import ask_sdk_core.utils as ask_utils
from ask_sdk_core.dispatch_components import (AbstractRequestHandler)
from ask_sdk_core.handler_input import HandlerInput


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Welcome to PolyAlexa, you can now ask a question about Cal Poly SLO."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )
