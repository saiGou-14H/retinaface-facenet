from flask import Flask
from app.app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001,debug=False,threaded =True)

