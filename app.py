from flask import Flask, render_template, jsonify
import os
import signal

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """
    Endpoint to kill the server process, useful to save runtime if needed.
    """
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"status": "shutting down..."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
