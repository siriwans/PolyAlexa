from flask import Flask, render_template
from flask_ask import Ask, statement, question, session

ASK_ROUTE = '/'
ask = Ask(route=ASK_ROUTE)

@ask.launch
def launch():
    speech_text = "Welcome to PolyAlexa, you can now ask a question about Cal Poly SLO."
    return question(speech_text).reprompt(speech_text).simple_card('LaunchIntent', speech_text)

@ask.intent('QuestionIntent')
def answer_question(question):
    speech_text = 'Question: ' + question
    return statement(speech_text).simple_card('QuestionIntent', speech_text)


@ask.intent('AMAZON.HelpIntent')
def help():
    speech_text = 'You can say ask me anything about Cal Poly SLO!'
    return question(speech_text).reprompt(speech_text).simple_card('HelpIntent', speech_text)


@ask.intent('AMAZON.StopIntent')
def stop():
    bye_text = "Goodbye!"
    return statement(bye_text)


@ask.intent('AMAZON.CancelIntent')
def cancel():
    bye_text = "Goodbye!"
    return statement(bye_text)


@ask.session_ended
def session_ended():
    return "{}", 200


def create_app():
    app = Flask(__name__)
    ask.init_app(app)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)


'''# --------------- Helpers that build all of the responses ----------------------

def build_speechlet_response(output, reprompt_text, should_end_session):
    return {
        'outputSpeech': {
            'type': 'PlainText',
            'text': output
        },
        'reprompt': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': reprompt_text
            }
        },
        'shouldEndSession': should_end_session
    }

def build_response(session_attributes, speechlet_response):
    return {
        'version': '1.0',
        'sessionAttributes': session_attributes,
        'response': speechlet_response
    }


# --------------- Functions that control the skill's behavior ------------------
def get_question_response(intent):
    """ An example of a custom intent. Same structure as welcome message, just make sure to add this intent
    in your alexa skill in order for it to work.
    """
    session_attributes = {}
    print(intent)
    question = intent['slots']['question']['value']

    # TODO generate answer to question here.....

    speech_output = "The question is: " + question
    reprompt_text = "Please say a question."
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        speech_output, reprompt_text, should_end_session))

def get_welcome_response():
    """ If we wanted to initialize the session to have some attributes we could
    add those here
    """
    session_attributes = {}
    speech_output = "Welcome to PolyAlexa, you can now ask a questio about Cal Poly SLO."

    # If the user either does not reply to the welcome message or says something
    # that is not understood, they will be prompted again with this text.

    reprompt_text = "I don't know if you heard me, welcome to PolyAlexa. You can now ask a question about Cal Poly SLO."
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        speech_output, reprompt_text, should_end_session))


def handle_session_end_request():
    speech_output = "Thank you for using PolyAlexa. Have a good day!"

    # Setting this to true ends the session and exits the skill.
    should_end_session = True
    return build_response({}, build_speechlet_response(
        speech_output, None, should_end_session))

# --------------- Events ------------------

def on_session_started(session_started_request, session):
    """ Called when the session starts.
        One possible use of this function is to initialize specific
        variables from a previous state stored in an external database
    """
    # Add additional code here as needed
    # if we need to store any variables the user ask about...
    pass



def on_launch(launch_request, session):
    """ Called when the user launches the skill without specifying what they
    want
    """
    # Dispatch to your skill's launch message
    return get_welcome_response()


def on_intent(intent_request, session):
    """ Called when the user specifies an intent for this skill """

    intent = intent_request['intent']
    intent_name = intent_request['intent']['name']

    # Dispatch to your skill's intent handlers
    if intent_name == "QuestionIntent":
        return get_question_response(intent)
    elif intent_name == "AMAZON.HelpIntent":
        return get_welcome_response()
    elif intent_name == "AMAZON.CancelIntent" or intent_name == "AMAZON.StopIntent":
        return handle_session_end_request()
    else:
        raise ValueError("Invalid intent")


def on_session_ended(session_ended_request, session):
    """ Called when the user ends the session.
    Is not called when the skill returns should_end_session=true
    """
    print("on_session_ended requestId=" + session_ended_request['requestId'] +
          ", sessionId=" + session['sessionId'])
    # add cleanup logic here


# --------------- Main handler ------------------

def lambda_handler(event, context):
    """ Route the incoming request based on type (LaunchRequest, IntentRequest,
    etc.) The JSON body of the request is provided in the event parameter.
    """
    print("Incoming request...")

    """
    Uncomment this if statement and populate with your skill's application ID to
    prevent someone else from configuring a skill that sends requests to this
    function.
    """
    # if (event['session']['application']['applicationId'] !=
    #         "amzn1.echo-sdk-ams.app.[unique-value-here]"):
    #     raise ValueError("Invalid Application ID")

    if event['session']['new']:
        on_session_started({'requestId': event['request']['requestId']},
                           event['session'])

    if event['request']['type'] == "LaunchRequest":
        return on_launch(event['request'], event['session'])
    elif event['request']['type'] == "IntentRequest":
        return on_intent(event['request'], event['session'])
    elif event['request']['type'] == "SessionEndedRequest":
        return on_session_ended(event['request'], event['session'])'''
