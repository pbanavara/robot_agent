import base64
import json
import logging
import time

import requests

from nearai.agents.environment import Environment  # type: ignore

from c import (
    DEBUG,
    ASSISTANT_ROLE,
    LATEST_SCREENSHOT_FILE,
    SCREENSHOT_FILE_PATTERN,
    PNG_EXT,
    JPG_EXT,
    INDEX_HTML_FILE,
)
from client import CruxClient
from models import ChatMessage, Transition, AgentState, ConversationHistory
from render import (
    build_markdown_for_latest_transition,
    build_html_ui,
    create_random_cat_image_url,
)

env: Environment


def drop_pydev_debug():
    if not DEBUG:
        return
    import pydevd_pycharm  # type: ignore

    pydevd_pycharm.settrace(
        "localhost", port=12345, stdoutToServer=True, stderrToServer=True
    )


class Agent:
    def __init__(self, env: Environment, thread_id: str):
        self.env = env
        self.thread_id = thread_id
        self.state = AgentState.load(env, thread_id)
        self.history = ConversationHistory.load(env, thread_id)
        self.client = CruxClient(self.state.endpoint, env=self.env)

    def set_endpoint(self, endpoint: str):
        self.state.set_endpoint(self.env, endpoint)
        self.client = CruxClient(endpoint, env=self.env)

    def handle_command(self, query: str) -> str:
        

        if query.startswith("!cats"):
            sleep_time = 3
            parts = query.split(" ")
            if len(parts) > 1:
                try:
                    sleep_time = int(parts[1])
                except ValueError:
                    pass

            t0 = Transition(
                screenshot=create_random_cat_image_url(),
                action="goto('https://cats.com')",
                reasoning="we need more cats",
                chat_messages=[],
                done=False,
            )
            self.process_transition(t0)
            time.sleep(sleep_time)

            chat_messages = [
                ChatMessage(
                    role=ASSISTANT_ROLE,
                    message="Cats are coming",
                    timestamp="2025-01-01 00:00:10",
                )
            ]

            t1 = Transition(
                screenshot=create_random_cat_image_url(),
                action="click('button#more-cats')",
                reasoning="clicking button, receive cats",
                chat_messages=list(chat_messages),
                done=False,
            )
            self.process_transition(t1)
            time.sleep(sleep_time)
            chat_messages.append(
                ChatMessage(
                    role=ASSISTANT_ROLE,
                    message="Got you more cats",
                    timestamp="2025-01-01 00:00:20",
                )
            )

            t2 = Transition(
                screenshot=create_random_cat_image_url(),
                action="click('button#more-cats')",
                reasoning="is this Istanbul? it's cats everywhere",
                chat_messages=list(chat_messages),
                done=True,
            )
            self.process_transition(t2)
            time.sleep(sleep_time)
            return ""

        if query.startswith("!legacy"):
            self.state.use_native_rendering = False
            self.state.save(self.env)
            self.env.add_reply("Switched to legacy HTML rendering mode.")
            return ""

        return query

    def get_relevant_memories(self, query: str) -> list[str]:
        memories = json.loads(self.env.query_user_memory(query))
        memories = memories[:1]  # We can't delete so they repeat a lot
        self.env.add_system_log(f"Memories retrieved for '{query}': {memories}")
        return memories

    def save_screenshot(self, screenshot: str) -> str:
        # If the screenshot string is not base64, assume it is already a relative file path/URL.
        if not screenshot.startswith("data:image/"):
            return screenshot
        meta, base64_data = screenshot.split(",", 1)
        ext = PNG_EXT if "png" in meta else JPG_EXT
        img_bytes = base64.b64decode(base64_data)
        self.state.increment_screenshot_counter(self.env)
        filename = SCREENSHOT_FILE_PATTERN.format(
            counter=self.state.screenshot_counter, ext=ext
        )
        self.env.write_file(filename, img_bytes, filetype=f"image/{ext}")
        self.env.write_file(
            LATEST_SCREENSHOT_FILE, filename.encode(), filetype="text/plain"
        )
        return filename

    def process_transition(self, transition: Transition):
        image = transition.screenshot
        if image:
            transition.screenshot = self.save_screenshot(image)

        if transition.done and transition.action == "wait_for_user_message()":
            if not transition.chat_messages:
                self.env.add_reply("The agent is waiting for your next instruction.")

        self.history.add_transition(transition)
        self.history.save(self.env)
        self.refresh_ui(image)

    def refresh_ui(self, last_image: str | None):
        if self.state.use_native_rendering:
            rendered = build_markdown_for_latest_transition(
                self.history.transitions, last_image
            )
            self.env.add_reply(rendered)
        else:
            rendered = build_html_ui(self.history.transitions, last_image)
            self.env.write_file(
                INDEX_HTML_FILE, rendered.encode(), filetype="text/html"
            )

    def _get_last_user_message(self) -> str:
        msgs = self.env.list_messages()
        user_msgs = [m for m in msgs if m["role"] != ASSISTANT_ROLE]
        return user_msgs[-1]["content"] if user_msgs else ""

    def consume_latest_message(self):
        if not self.client.health_check():
            self.env.add_reply(
                f"Cannot connect to {self.state.endpoint}. Server appears down."
            )
            return

        query = self._get_last_user_message()
        query = self.handle_command(query).strip()
        if not query:
            return

        memories = self.get_relevant_memories(query)

        try:
            for transition in self.client.stream_transitions(
                query, self.thread_id, memories
            ):
                self.process_transition(transition)
        except requests.exceptions.ConnectionError as e:
            error_msg = (
                f"Cannot connect to the AI server at {self.client.endpoint}. "
                f"The server appears to be down or not running. "
                f"Please try again later."
            )
            self.env.add_system_log(f"{error_msg}\nDetails: {e}", logging.ERROR)
            if DEBUG:
                drop_pydev_debug()
            self.env.add_reply(f"Connection failed with error: {e}")
        except Exception as e:
            self.env.add_system_log(f"Error processing request: {e}", logging.ERROR)
            if DEBUG:
                drop_pydev_debug()
            self.env.add_reply(f"Agent failed with error: {e}")
        self.env.add_system_log("Finished processing latest message")


def run(env: Environment):
    env._debug_mode = True
    agent = Agent(env, env._thread_id)

    if DEBUG:
        # agent.set_endpoint("http://localhost:8003")
        import pydevd_pycharm  # type: ignore

        pydevd_pycharm.settrace(
            "localhost", port=12345, stdoutToServer=True, stderrToServer=True
        )

    env.add_system_log(f"Initialized agent thread: {env._thread_id}")
    agent.consume_latest_message()


run(env)
