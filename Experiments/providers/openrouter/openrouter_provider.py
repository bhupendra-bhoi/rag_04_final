"""OpenRouter provider strategy.

OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible Chat Completions
API, so this provider uses the official ``openai`` SDK pointed at OpenRouter's
base URL. It mirrors ``GroqProvider`` exactly for multi-key rotation: keys are
supplied as a list (resolved from a comma-separated ``OPENROUTER_API_KEY`` env
var by the factory), and on a 429 the current key is put on cooldown (honoring
the response's ``retry-after`` / ``x-ratelimit-reset`` hints) before rotating to
the next available key. When all keys are cooling down, AllKeysExhaustedException
is raised — identical semantics to Groq so the experiment runner's retry logic
behaves the same.
"""

from datetime import datetime, timedelta

from openai import (
    OpenAI,
    RateLimitError,
    APIStatusError,
    APITimeoutError,
    APIConnectionError,
)

from providers.base.strategy import ProviderStrategy
from providers.registry import provider_registry
from providers.enums import ProviderType
from providers.base.key_state import KeyState
from providers.base.exceptions import (
    AllKeysExhaustedException,
    ProviderInitializationException,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@provider_registry.register(ProviderType.OPENROUTER)
class OpenRouterProvider(ProviderStrategy):

    DEFAULT_COOLDOWN_SECONDS = 60

    # HTTP statuses where rotating to another key (and cooling this one) may help.
    #   429 rate limit, 402 out-of-credits, 408 timeout, 409 conflict,
    #   425 too-early, 5xx upstream/provider errors, 529 overloaded.
    # (429 is handled by the more specific RateLimitError branch; listed for clarity.)
    RETRYABLE_STATUS = {402, 408, 409, 425, 429, 500, 502, 503, 504, 529}

    def __init__(
        self,
        api_keys: list[str],
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        config=None,
        base_url: str = DEFAULT_BASE_URL,
    ):
        super().__init__(config)

        if not api_keys:
            raise ProviderInitializationException(
                "At least one API key is required."
            )

        # Allow base_url and optional OpenRouter ranking headers to be overridden
        # via provider params without changing code.
        params = getattr(config, "params", None) or {}
        base_url = params.get("base_url", base_url)
        self._extra_headers = {}
        if params.get("http_referer"):
            self._extra_headers["HTTP-Referer"] = params["http_referer"]
        if params.get("x_title"):
            self._extra_headers["X-Title"] = params["x_title"]

        self.cooldown_seconds = cooldown_seconds
        self.current_index = 0
        self.keys = [KeyState(api_key=key) for key in api_keys]
        self.clients = {}

        try:
            for key in api_keys:
                self.clients[key] = OpenAI(api_key=key, base_url=base_url)
        except Exception as exc:
            raise ProviderInitializationException(
                f"Failed to initialize OpenRouter clients: {exc}"
            ) from exc

        print(
            f"OpenRouter provider initialized with {len(self.keys)} key(s) "
            f"(rotation needs 2+ keys; set OPENROUTER_API_KEY comma-separated)."
        )

    def _mark_rate_limited(
        self,
        key_state: KeyState,
        retry_after_seconds: int | None = None,
    ):
        cooldown_seconds = (
            retry_after_seconds
            if retry_after_seconds is not None
            else self.cooldown_seconds
        )
        key_state.cooldown_until = (
            datetime.now() + timedelta(seconds=cooldown_seconds)
        )
        key_state.total_429s += 1

    @staticmethod
    def _retry_after_seconds(exc: RateLimitError):
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)

        if not headers:
            return None

        if retry := headers.get("retry-after"):
            try:
                return int(float(retry))
            except (TypeError, ValueError):
                pass

        # OpenRouter/OpenAI-style reset hints. x-ratelimit-reset is commonly an
        # epoch value in milliseconds; convert to a delay from now.
        for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset"):
            reset = headers.get(name)
            if not reset:
                continue
            reset = str(reset)
            if reset.endswith("ms"):
                return max(1, round(float(reset[:-2]) / 1000))
            if reset.endswith("s"):
                return int(float(reset[:-1]))
            try:
                val = float(reset)
            except ValueError:
                continue
            # Heuristic: large value => epoch (ms if very large), else a delay.
            if val > 1e12:  # epoch milliseconds
                return max(1, round(val / 1000 - datetime.now().timestamp()))
            if val > 1e9:   # epoch seconds
                return max(1, round(val - datetime.now().timestamp()))
            return max(1, int(val))

        return None

    def _get_next_available_key(self) -> KeyState:
        total_keys = len(self.keys)
        for offset in range(total_keys):
            idx = (self.current_index + offset) % total_keys
            key_state = self.keys[idx]
            if key_state.available:
                self.current_index = idx
                return key_state
        raise AllKeysExhaustedException(
            "All OpenRouter API keys are currently rate limited."
        )

    def generate(self, model: str, messages: list, **kwargs):
        attempted_keys = set()

        while len(attempted_keys) < len(self.keys):
            key_state = self._get_next_available_key()

            print(f"Using key #{self.current_index}: ****{key_state.api_key[-4:]}")

            if key_state.api_key in attempted_keys:
                break
            attempted_keys.add(key_state.api_key)

            client = self.clients[key_state.api_key]

            try:
                key_state.total_requests += 1

                create_kwargs = dict(model=model, messages=messages, **kwargs)
                if self._extra_headers:
                    create_kwargs["extra_headers"] = self._extra_headers

                response = client.chat.completions.create(**create_kwargs)

                # OpenRouter can return HTTP 200 with an error payload and no
                # usable choices (e.g. the upstream provider was rate-limited).
                # No exception is raised in that case, so detect it here and
                # rotate instead of letting the empty response crash downstream.
                err = getattr(response, "error", None)
                if err or not getattr(response, "choices", None):
                    key_state.total_failures += 1
                    print("=" * 80)
                    print(f"Empty/error 200 response on ****{key_state.api_key[-4:]}: {err}")
                    self._mark_rate_limited(key_state=key_state)
                    print(self.health())
                    continue

                key_state.total_successes += 1
                return response

            except RateLimitError as exc:
                # Explicit 429.
                key_state.total_failures += 1
                print("=" * 80)
                print(f"429 (rate limit) on ****{key_state.api_key[-4:]}")
                print(exc)
                retry_after = self._retry_after_seconds(exc)
                print("Retry after:", retry_after)
                self._mark_rate_limited(
                    key_state=key_state,
                    retry_after_seconds=retry_after,
                )
                print(self.health())

            except (APITimeoutError, APIConnectionError) as exc:
                # Transient network problem — brief cooldown, then rotate.
                key_state.total_failures += 1
                print("=" * 80)
                print(f"Connection/timeout on ****{key_state.api_key[-4:]}: {exc}")
                self._mark_rate_limited(key_state=key_state, retry_after_seconds=5)
                print(self.health())

            except APIStatusError as exc:
                # Other HTTP errors (402 out-of-credits, 5xx upstream, ...).
                # Rotate on retryable statuses; re-raise real errors (400/401/
                # 404/422) since rotating cannot fix a bad request or bad auth.
                status = getattr(exc, "status_code", None)
                if status in self.RETRYABLE_STATUS:
                    key_state.total_failures += 1
                    print("=" * 80)
                    print(f"HTTP {status} (retryable) on ****{key_state.api_key[-4:]}")
                    print(exc)
                    retry_after = self._retry_after_seconds(exc)
                    self._mark_rate_limited(
                        key_state=key_state,
                        retry_after_seconds=retry_after,
                    )
                    print(self.health())
                else:
                    key_state.total_failures += 1
                    raise

            except Exception:
                key_state.total_failures += 1
                raise

        raise AllKeysExhaustedException(
            "No available OpenRouter API keys (all rate-limited/cooling down)."
        )

    def health(self) -> dict:
        return {
            "provider": "openrouter",
            "current_index": self.current_index,
            "keys": [
                {
                    "key_suffix": key.api_key[-4:],
                    "available": key.available,
                    "cooldown_until": key.cooldown_until,
                    "requests": key.total_requests,
                    "successes": key.total_successes,
                    "failures": key.total_failures,
                    "429s": key.total_429s,
                }
                for key in self.keys
            ],
        }

    def current_key(self) -> dict:
        """Return information about the key that will be used next."""
        key = self.keys[self.current_index]
        return {
            "index": self.current_index,
            "key_suffix": key.api_key[-4:],
            "available": key.available,
            "cooldown_until": key.cooldown_until,
            "requests": key.total_requests,
            "successes": key.total_successes,
            "failures": key.total_failures,
            "429s": key.total_429s,
        }
