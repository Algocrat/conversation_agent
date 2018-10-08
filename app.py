from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort, Response, json

from predict import ConversationAgent

app = Flask(__name__)

conversation_agent = ConversationAgent()
conversations = []


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/reply', methods=['POST'])
def reply():
    if request.method == 'POST':
        content = request.json
        sentence = content['sentence']
        if not sentence:
            flash('No sentence post')
            redirect(request.url)
        elif len(sentence) == 0:
            flash('No sentence')
            redirect(request.url)
        else:
            conversations.append('YOU: ' + sentence)
            bot_reply = conversation_agent.reply(sentence)
            conversations.append('BOT: ' + bot_reply)
    return Response(json.dumps(conversations),  mimetype='application/json')

if __name__ == '__main__':
    app.run()

#curl -H 'Content-Type: application/json' -X POST -d '{"sentence":"Hey!How are you?"}' http://localhost:5000/reply
#curl -H 'Content-Type: application/json' -X POST -d '{"sentence":"Where are you?"}' http://localhost:5000/reply
#curl -H 'Content-Type: application/json' -X POST -d '{"sentence":"Who are you?"}' http://localhost:5000/reply
