#!/usr/bin/env python3
"""
本地大模型命令行 Chat 客户端
支持 OpenAI 兼容接口 / 流式 & 非流式输出 / <think> 标签独立展示

用法:
  python chat.py                        # 默认配置
  python chat.py --stream               # 流式输出
  python chat.py --no-stream            # 非流式输出
  python chat.py --url http://x:8000   # 自定义服务地址
  python chat.py --model Qwen2.5-7B    # 指定模型名
  python chat.py --system "你是一个助手"  # 自定义系统提示
"""

import argparse
import re
import sys

try:
    from openai import OpenAI
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
    from rich import print as rprint
except ImportError:
    print("缺少依赖，请先安装：pip install openai rich")
    sys.exit(1)

# ─────────────────────────── 默认配置 ───────────────────────────
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL    = ""
DEFAULT_SYSTEM   = "You are a helpful assistant."
DEFAULT_STREAM   = True
# ────────────────────────────────────────────────────────────────

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="本地大模型命令行 Chat 客户端")
    parser.add_argument("--url",       default=DEFAULT_BASE_URL, help="vLLM 服务地址")
    parser.add_argument("--model",     default=DEFAULT_MODEL,    help="模型名称")
    parser.add_argument("--system",    default=DEFAULT_SYSTEM,   help="系统提示词")
    parser.add_argument("--stream",    dest="stream", action="store_true",  default=DEFAULT_STREAM)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--max-tokens", type=int, default=4096,  help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


# ───────────────────────── <think> 解析 ─────────────────────────

def split_think_and_reply(text: str):
    """
    将文本拆分为 thinking 部分和 reply 部分。
    返回 (thinking: str | None, reply: str)
    """
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = pattern.search(text)
    if match:
        thinking = match.group(1).strip()
        reply = pattern.sub("", text).strip()
        return thinking, reply
    return None, text.strip()


def render_response(full_text: str):
    """非流式：渲染完整回复（含 think 分离）"""
    thinking, reply = split_think_and_reply(full_text)

    if thinking:
        console.print(
            Panel(
                thinking,
                title="[bold yellow]💭 思考过程[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    console.print(
        Panel(
            Markdown(reply),
            title="[bold green]🤖 Assistant[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


# ─────────────────────── 流式输出处理 ───────────────────────────

import sys

class StreamRenderer:
    """
    流式输出状态机：
      - 收集 <think>...</think> 期间实时展示思考内容（黄色）
      - 思考结束后切换为正常回复展示（绿色）
    """

    THINK_OPEN  = "<think>"
    THINK_CLOSE = "</think>"

    def __init__(self):
        self.buffer       = ""
        self.in_think     = False
        self.think_done   = False
        self.think_buf    = ""
        self.reply_buf    = ""
        self._think_header_printed = False
        self._reply_header_printed = False

    def _write(self, text: str, color: str = ""):
        """直接写入 stdout，避免 rich 解析 $ 和 <tag>"""
        if color == "yellow":
            sys.stdout.write(f"\033[33m{text}\033[0m")
        elif color == "green":
            sys.stdout.write(f"\033[32m{text}\033[0m")
        else:
            sys.stdout.write(text)
        sys.stdout.flush()

    def _ensure_think_header(self):
        if not self._think_header_printed:
            console.print()
            console.print(Rule("[bold yellow]💭 思考过程[/bold yellow]", style="yellow"))
            self._think_header_printed = True

    def _ensure_reply_header(self):
        if not self._reply_header_printed:
            console.print()
            console.print(Rule("[bold green]🤖 Assistant[/bold green]", style="green"))
            self._reply_header_printed = True

    def feed(self, chunk: str):
        self.buffer += chunk

        while self.buffer:
            if not self.in_think and not self.think_done:
                idx = self.buffer.find(self.THINK_OPEN)
                if idx == -1:
                    # 没有 <think>，检查是否是 <think> 的不完整前缀
                    # 防止 "<thi" 这种截断块被误输出
                    safe_len = self._safe_output_len(self.buffer, self.THINK_OPEN)
                    if safe_len > 0:
                        out = self.buffer[:safe_len]
                        self._ensure_reply_header()
                        self._write(out)
                        self.reply_buf += out
                        self.buffer = self.buffer[safe_len:]
                    break  # 等待更多数据
                else:
                    before = self.buffer[:idx]
                    if before:
                        self._ensure_reply_header()
                        self._write(before)
                        self.reply_buf += before
                    self.buffer = self.buffer[idx + len(self.THINK_OPEN):]
                    self.in_think = True
                    self._ensure_think_header()

            elif self.in_think:
                idx = self.buffer.find(self.THINK_CLOSE)
                if idx == -1:
                    safe_len = self._safe_output_len(self.buffer, self.THINK_CLOSE)
                    if safe_len > 0:
                        out = self.buffer[:safe_len]
                        self._write(out, color="yellow")
                        self.think_buf += out
                        self.buffer = self.buffer[safe_len:]
                    break
                else:
                    tail = self.buffer[:idx]
                    if tail:
                        self._write(tail, color="yellow")
                        self.think_buf += tail
                    self.buffer = self.buffer[idx + len(self.THINK_CLOSE):]
                    self.in_think  = False
                    self.think_done = True
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            else:
                # 思考结束，正常输出 reply
                self._ensure_reply_header()
                self._write(self.buffer)
                self.reply_buf += self.buffer
                self.buffer = ""

    def _safe_output_len(self, buf: str, tag: str) -> int:
        """
        返回可以安全输出的字符数。
        若 buf 末尾是 tag 的某个前缀，则保留该前缀不输出，等待后续数据。
        """
        for i in range(min(len(tag) - 1, len(buf)), 0, -1):
            if buf.endswith(tag[:i]):
                return len(buf) - i
        return len(buf)

    def finalize(self):
        # 输出缓冲区剩余内容
        if self.buffer:
            if self.in_think:
                self._write(self.buffer, color="yellow")
                self.think_buf += self.buffer
            else:
                self._ensure_reply_header()
                self._write(self.buffer)
                self.reply_buf += self.buffer
            self.buffer = ""

        sys.stdout.write("\n")
        sys.stdout.flush()

        console.print()


# ──────────────────────────── 主逻辑 ────────────────────────────

def chat_stream(client, model, messages, max_tokens, temperature):
    renderer = StreamRenderer()
    try:
        with client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    renderer.feed(delta)
        renderer.finalize()
        return renderer.think_buf, renderer.reply_buf
    except Exception as e:
        console.print(f"\n[bold red][ERROR] 流式请求失败: {e}[/bold red]")
        return "", ""


def chat_no_stream(client, model, messages, max_tokens, temperature):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        full_text = resp.choices[0].message.content or ""
        render_response(full_text)
        thinking, reply = split_think_and_reply(full_text)
        return thinking or "", reply
    except Exception as e:
        console.print(f"\n[bold red][ERROR] 请求失败: {e}[/bold red]")
        return "", ""


def print_welcome(args):
    console.print(
        Panel(
            f"[bold]服务地址[/bold]  {args.url}\n"
            f"[bold]模型名称[/bold]  {args.model}\n"
            f"[bold]输出模式[/bold]  {'流式 (streaming)' if args.stream else '非流式 (non-streaming)'}\n"
            f"[bold]系统提示[/bold]  {args.system}\n\n"
            "[dim]输入 [bold]exit[/bold] 或 [bold]quit[/bold] 退出，[bold]clear[/bold] 清空上下文[/dim]",
            title="[bold cyan]🚀 本地大模型 Chat[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def main():
    args   = parse_args()
    client = OpenAI(base_url=args.url, api_key="EMPTY")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print_welcome(args)

    while True:
        try:
            console.print("[bold cyan]You >[/bold cyan] ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]已退出。[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            console.print("[dim]已退出。[/dim]")
            break

        if user_input.lower() == "clear":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            console.print("[dim]✅ 上下文已清空。[/dim]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        if args.stream:
            _, reply = chat_stream(
                client, args.model, messages, args.max_tokens, args.temperature
            )
        else:
            _, reply = chat_no_stream(
                client, args.model, messages, args.max_tokens, args.temperature
            )

        if reply:
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

