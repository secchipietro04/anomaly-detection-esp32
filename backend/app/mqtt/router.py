# fastapi-style mqtt router and topic pattern matcher
import re
import inspect
from typing import Callable, Dict, List, Tuple, Any, Optional

class MQTTRoute:
    # encapsulates single subscribed topic pattern and its handler
    def __init__(self, pattern: str, handler: Callable[..., Any]):
        self.pattern = pattern
        self.handler = handler
        self.param_names, self.regex = self._compile_pattern(pattern)

    def _compile_pattern(self, pattern: str) -> Tuple[List[str], re.Pattern]:
        # transform v1/{node_id}/data/sensor into regex pattern
        param_names = []
        parts = pattern.split("/")
        regex_parts = []
        for part in parts:
            if part.startswith("{") and part.endswith("}"):
                p_name = part[1:-1]
                param_names.append(p_name)
                regex_parts.append(r"([^/]+)")
            elif part == "+":
                param_names.append("_wildcard")
                regex_parts.append(r"([^/]+)")
            elif part == "#":
                regex_parts.append(r"(.*)")
            else:
                regex_parts.append(re.escape(part))
        regex = re.compile("^" + "/".join(regex_parts) + "$")
        return param_names, regex

    def match(self, topic: str) -> Optional[Dict[str, Any]]:
        # match topic string and extract path arguments
        m = self.regex.match(topic)
        if not m:
            return None
        args = {}
        for name, val in zip(self.param_names, m.groups()):
            if name != "_wildcard":
                args[name] = val
        return args

class MQTTRouter:
    # collection of routes with decorator interface
    def __init__(self, prefix: str = "v1"):
        self.prefix = prefix
        self.routes: List[MQTTRoute] = []

    def subscribe(self, topic_pattern: str):
        # decorator registering topic pattern handler
        def decorator(func: Callable[..., Any]):
            pattern = topic_pattern
            if not pattern.startswith("/") and not pattern.startswith(self.prefix) and not pattern.startswith("+"):
                pattern = f"{self.prefix}/{pattern}"
            route = MQTTRoute(pattern, func)
            self.routes.append(route)
            return func
        return decorator

    def resolve(self, topic: str) -> Optional[Tuple[Callable[..., Any], Dict[str, Any]]]:
        # find matching handler and extracted args for topic
        for route in self.routes:
            args = route.match(topic)
            if args is not None:
                return route.handler, args
        return None

    def get_subscription_topics(self) -> List[str]:
        # return mqtt wildcard patterns for broker subscription
        subs = set()
        for r in self.routes:
            # convert {param} to + for mqtt subscribe
            parts = [("+" if (p.startswith("{") and p.endswith("}")) else p) for p in r.pattern.split("/")]
            subs.add("/".join(parts))
        return list(subs)
