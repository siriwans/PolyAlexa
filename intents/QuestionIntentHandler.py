import ask_sdk_core.utils as ask_utils
from ask_sdk_core.dispatch_components import (AbstractRequestHandler)
from ask_sdk_core.handler_input import HandlerInput

class QuestionIntentHandler(AbstractRequestHandler):
    """Handler for Question Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("QuestionIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        slot = ask_utils.request_util.get_slot(handler_input, "question")
        question = slot.value

        speak_output = "Question: " + question

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )
