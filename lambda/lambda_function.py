from flask import Flask
from flask_ask import Ask, statement, question
from .predict1 import *

ASK_ROUTE = '/'
ask = Ask(route=ASK_ROUTE)

@ask.launch
def launch():
    speech_text = "Welcome to PolyAlexa, you can now ask a question about Cal Poly SLO."
    return question(speech_text).reprompt(speech_text).simple_card('LaunchIntent', speech_text)

@ask.intent('QuestionIntent')
def answer_question(ques):
    # who is the president of cal pol
    answer = get_answer(ques)
    print("ANSWER:", answer)
    speech_text = ''
    if answer:
        speech_text = str(answer)
    else:
        speech_text = 'I can\'t seem to find the answer'
    return question(speech_text).reprompt(speech_text).simple_card('QuestionIntent', speech_text)


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
