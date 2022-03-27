import array
import os
import socket
import sys
import numpy as np
import ast

import cv2
import warnings

import scratch
import threading

class UnityServer():

    def __init__(self,port=10000):
        self.done = False
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_address = ('localhost', 10000)
        print('starting up on %s port %s' % self.server_address)
        self.sock.bind(self.server_address)
        # Listen for incoming connections
        self.sock.listen(1)
        print('waiting for a connection')
        self.connection, self.client_address = self.sock.accept()
        print("sucsessfull conection from " + str(self.client_address))

        self.state = (None,None)
        self.action= (0,0)

        self.imgs2Stitch : list = []
        self.recivingStitch = True




    def update(self, BUFFER_SIZE=2 ** 15):

        loc = None
        img_np = None
        while not self.done:
            data = self.connection.recv(BUFFER_SIZE)
            data = array.array('B', data)
            try:
                t = bytes(data).decode()
                print(t[0])
            except UnicodeDecodeError:
                continue
            if t[0] == 'l':
                self.connection.sendall("expecting to receive loc".encode())
                dataIn = self.connection.recv(BUFFER_SIZE)
                loc = ast.literal_eval(dataIn.decode())
                self.connection.sendall("ack Loc".encode())
                #print('recived' + str(loc))

            #case where picture is sent if ps then we are in stitching mode
            #if p is sent then normal generic state update is done
            elif t[0] == 'p' or t[0] == 's':
                #ps must be sent first if not sent no stitching
                if t[0] == 'p':
                    self.recivingStitch = False
                size = int(t.split()[1])
                if self.action is 'reset' or self.action is 'done':
                    resp = str(self.action)#'expecting image of size ' + str(size)
                else:
                    resp = str(list(self.action))#'expecting image of size ' + str(size)
                #print(resp)
                self.connection.sendall(resp.encode())
                picIn = self.connection.recv(size)
                picIn= np.fromstring(picIn, np.uint8)
                #note read by cv means we are in BGR color
                iraw = cv2.imdecode(picIn, flags=1)
                if iraw is not None:
                    img_np = iraw
                    if self.recivingStitch:
                        self.imgs2Stitch.append(img_np)

                else:
                    warnings.warn("misformated image sent not updating state")
                #if img_np is not None:
                    # cv2.imshow('test',img_np)
                    # cv2.waitKey(1)
                #cv2.destroyAllWindows()

                self.connection.sendall("ack IMG".encode())

            self.state = (loc,img_np)



    def dumpPhotos(self,path,BUFFER_SIZE=2**15):
        img_np = None
        count = 0
        while not self.done:
            data = self.connection.recv(BUFFER_SIZE)
            data = array.array('B', data)
            try:
                t = bytes(data).decode()
            except UnicodeDecodeError:
                continue
            if t[0] == 'p':
                size = int(t.split()[1])
                if self.action is 'reset':
                    resp = str(self.action)  # 'expecting image of size ' + str(size)
                else:
                    resp = str(list(self.action))  # 'expecting image of size ' + str(size)
                # print(resp)
                self.connection.sendall(resp.encode())
                picIn = self.connection.recv(size)
                picIn = np.fromstring(picIn, np.uint8)
                iraw = cv2.imdecode(picIn, flags=1)
                if iraw is not None:
                    img_np = iraw
                else:
                    warnings.warn("misformated image sent not updating state")
                if img_np is not None:
                    cv2.imshow('test',img_np)
                    cv2.waitKey(1)
                    cv2.imwrite(os.path.join(path,'test'+str(count)+".png"),img_np)
                    #cv2.destroyAllWindows()
                    print(count)
                    count =count+1

                self.connection.sendall("ack IMG".encode())




if __name__ == '__main__':





    s = UnityServer()


    #s.update()
    # path = os.path.join("c:",os.sep,"Users","samiw","OneDrive","Desktop","Desktop","VT","Research","imageStitch","testImages5")
    # s.dumpPhotos(path)

    servThread = threading.Thread(target=s.update)
    servThread.start()

    while True:
        img = s.state[1]


