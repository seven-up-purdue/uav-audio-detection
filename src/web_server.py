import os
from datetime import datetime

import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.gen

import json

# WEB_SERVER_ADDRESS = ('localhost', 8090)
WEB_SERVER_ADDRESS = ('0.0.0.0', 8090)

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')

# all clients - browsers and rpis
clients = []
# only raspberry pis
rpis = []
# only browsers
browsers = []

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        """
        add newly connected clients to client lists
        :return: 
        """
        print("New connection")
        clients.append(self)
        device = self.get_argument("device")
        if device == "rpi":
            print("Raspberry Pi connected")
            self.write_message("rpi hello")
            rpis.append(self)
        elif device == "browser":
            print("Something else")
            browsers.append(self)

    #
    def on_message(self, message): # received from rpis
        """
        Transfer an incoming message to the browsers
        :param message: 
        :return: 
        """
        print("message: %s" % message)
        for b in browsers:
            b.write_message(message)


    def on_close(self):
        print("Connection closed")
        clients.remove(self)
        device = self.get_argument("device")
        if device == "rpi":
            print("Raspberry Pi closed")
            rpis.remove(self)
        elif device == "browser":
            print("Browser connection closed")
            browsers.remove(self)


def main():
    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "static"),
    }
    app = tornado.web.Application(
        handlers=[
            (r"/", IndexHandler),
            (r"/ws", WebSocketHandler),
        ], **settings
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(WEB_SERVER_ADDRESS[1], WEB_SERVER_ADDRESS[0])
    print("Listening on port:", WEB_SERVER_ADDRESS[1])
 
    main_loop = tornado.ioloop.IOLoop.instance()
    main_loop.start()
 
if __name__ == "__main__":
    print("web server start")
    main()

