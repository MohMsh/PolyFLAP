import hashlib
import pickle
import random
import socket
import secrets
import string
import threading
import time
from copy import deepcopy
import local_model

import aes
from sklearn import metrics


class Client:
    """
    this class initiates an instance of the client class
    the functions in this class are used to create a socket, connect to server and launch the training process
    also, there is a function to close the socket after finishing
    """

    def __init__(self, host, port, buffer_size, receive_timeout, print_incoming_messages,
                 print_sent_messages, x_train, y_train, x_test, y_test):
        """
        class constructor
        :param host: IP of server
        :param port: port of application
        :param buffer_size: size of exchanged message
        :param receive_timeout: time before client drops connection if no response from server
        """
        self.initial_key = None
        self.print_sent_messages = print_sent_messages
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.receive_timeout = receive_timeout
        self.print_incoming_messages = print_incoming_messages
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.receive_start_time = None
        self.my_socket = None
        self.random_secret = None

    def create_socket(self):
        """
        this function creates the socket for the connection using the IP4 and socket stream
        :return: None
        """
        self.my_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        print("\nClient socket created")
        self.print_line()

    def connect(self):
        """
        this function connects the client to the server using the IP and port provided
        :return: None
        """
        try:
            self.my_socket.connect((self.host, self.port))
            print("Client connected to server  successfully")
            self.print_line()

            # generate a 32 random_key to randomize the initial key generation
            alphabet = string.ascii_letters + string.digits
            self.random_secret = ''.join(secrets.choice(alphabet) for i in range(32))

            # the initial key will be generated using this randomly created secret
            # later, the secret will be shuffled and sent to client with the connect message
            self.initial_key = self.generate_initial_key(self.my_socket.getsockname(), self.random_secret)

            return True

        except BaseException as e:
            print("Error Connecting to the Server: {msg}.".format(msg=e))
            self.print_line()
            return False

    def join_cycle(self):
        """
        this function launches the training method on a thread. Knowing that the training is not the first step,
        the steps followed in this function are as follows:
            1. Connect to server and send a "Connect" message
            2. Receive the EncToKs and send a "Ready" message
            3. Receive the model and start the training
        :return: None
        """
        exchange_messages_thread = ExchangeMessagesThread(buffer_size=self.buffer_size,
                                                          receive_timeout=self.receive_timeout,
                                                          this_socket=self.my_socket, initial_key=self.initial_key,
                                                          receive_start_time=time.time(),
                                                          print_incoming_messages=self.print_incoming_messages,
                                                          print_sent_messages=self.print_sent_messages,
                                                          x_train=self.x_train, y_train=self.y_train,
                                                          x_test=self.x_test, y_test=self.y_test,
                                                          random_secret=self.random_secret)
        exchange_messages_thread.start()

    def close_socket(self):
        """
        this function closes the socket after finishing the exchange
        :return: None
        """
        self.my_socket.close()
        print("\n\t Socket closed")

    @staticmethod
    def print_line():
        print("=======================================================================\n")

    def generate_initial_key(self, client_address, random_secret):
        # generate a key which combines both address and secret
        # take first 4 and last 4 characters from the address
        # take characters [0:8], [12:20], [24:32] characters from secret

        # build the key by merging the 32 characters where:
        # the reverse of the third substring from secret become characters [0:8]
        # the second substring from address becomes characters from [8:12]
        # the second substring from secret become characters [12:20]
        # the first substring from address becomes characters [20:24]
        # the reverse of the first substring from secret becomes characters [8:24]
        merged_secret = f"{random_secret[24:][::-1]}{str(client_address)[-4:]}{random_secret[12:20]}{str(client_address)[:4]}{random_secret[:8][::-1]}"

        # If, for any reason, the resulting merged secret is not 32 characters, pad it wih "!"
        # until it reaches the desired length
        while True:
            if len(merged_secret) < 32:
                merged_secret += "!"
            else:
                break

        # encode the merged secret with 'utf-8' encoding
        address_bytes = merged_secret.encode('utf-8')

        # Generate the hash value using SHA-256
        hash_obj = hashlib.sha256(address_bytes)
        hash_bytes = hash_obj.digest()

        # Convert the hash bytes to a hex string
        hash_str = hash_bytes.hex()

        # Trim and take only the first 32 characters if the size of the key is more than 32 characters for any reason
        hash_str32 = hash_str[:32]

        # Return the first 16 characters of the hex string
        return hash_str32.encode('utf-8')


class ExchangeMessagesThread(threading.Thread):
    """
    this class defines the thread that handles communication with server
    """

    def __init__(self, buffer_size, receive_timeout, this_socket, receive_start_time, initial_key,
                 print_incoming_messages, print_sent_messages, x_train, y_train, x_test, y_test, random_secret):
        """
        class constructor
        :param buffer_size: size of exchanged message
        :param receive_timeout: time before client drops connection if no response from server
        :param this_socket: socket handling connection with server
        :param initial_key: global key used in AES ciphering
        """
        threading.Thread.__init__(self)
        self.print_sent_messages = print_sent_messages
        self.print_incoming_messages = print_incoming_messages
        self.receive_timeout = receive_timeout
        self.receive_start_time = receive_start_time
        self.buffer_size = buffer_size
        self.socket = this_socket
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.connection_status = 0
        self.message_subject = None
        self.message_content = None
        self.response_subject = None
        self.response_content = None
        self.response = None
        self.all_data_received_flag = None
        self.message_buffer = b""
        self.temp_buffer = b""
        self.received_message = None
        self.decrypted_message = None
        self.DecToKs = None
        self.EncToKs = None
        self.initial_key = initial_key
        self.random_secret = random_secret
        self.local_model = None

    def run(self):
        """
        this function runs the receiving thread to communicate with server
        :return:
        """

        self.response_subject = "Connect"
        client_name = input("\tEnter your name here please: ")

        # shuffle the random secret generated at the connect step by applying the below
        """
        1. slice the string to 4 substrings of 8 characters.
        2. characters 0 => 8 moved to 8  => 16
        3. characters 8 => 16 moved to 24 => 32
        4. characters 16 => 24 are reversed and become 0 => 8
        5. characters 24 => 32 are reversed and become 16 => 24
        """

        shuffled_secret = self.random_secret[16:24][::-1] + self.random_secret[0:8] + \
                          self.random_secret[24:][::-1] + self.random_secret[8:16]

        self.response_content = f"{client_name};{shuffled_secret}"
        self.send()

        while True:
            """
            to ensure more than one message is exchanged using the same socket, this while loop is used.
            In this loop, the client first checks if it is already connected to the socket or not.
                1. if not, connection is initiated and client identify himself to the server
                2. when connected, the client receive the EncToKs and send a message to state it is ready
                2. when the server receive the "Ready" message, it send the model and training starts 
            """

            # flush all variables to reset the messages exchange cycle
            self.received_message = None
            self.connection_status = 0

            # call the 'receive' function to receive message from server
            print("\n\t<===> Awaiting server's response :")
            self.received_message, self.connection_status = self.receive()

            if self.connection_status == 0:
                self.socket.close()
                print("\n\t /\\/ Connection closed with server \\/\\")
                break

            # call the reply function to reply to server based on the received message
            if not self.reply():
                print("\n\t /\\/ Connection closed with server \\/\\")
                break

    def reply(self):
        """
        This function handles sending the messages to the server
        :return:
        """
        """
            when the message is received, it should be broken down. If the
            message starts with an Index key, then it is encrypted, and we will
            need to decrypt it before being able to use it, else we can skip decryption
            (this is the case when the message is the EncToKs)
        """
        # print the received message if the printing option is activated
        if self.print_incoming_messages:
            print("\n\tThe message received: ", self.received_message)

        if "Index" in self.received_message:
            # First we decrypt the message using the key with the associated index
            self.decrypted_message = self.decrypt(key=self.DecToKs[self.received_message["Index"]][1],
                                                  message=self.received_message)
            self.message_subject = self.decrypted_message["subject"]
            self.message_content = self.decrypted_message["content"]

            # print the decrypted received message if the printing option is activated
            if self.print_incoming_messages:
                print("\tThe decrypted version is: ", self.decrypted_message)
        else:
            self.message_subject = self.received_message["subject"]
            self.message_content = self.received_message["content"]

        # if the message is entitled as 'EncToKs', store the encrypted table of keys, decrypt it
        # and reply to server with a 'Ready' message so the training can start
        if self.message_subject == "EncToKs":
            self.EncToKs = self.message_content

            # create a blank list and assign it to DecToKs to hold the decrypted table of keys
            self.DecToKs = None
            self.DecToKs = [[] for x in range(len(self.EncToKs))]  # create the variable

            # decrypt the received table of keys using the master key
            # Till now, exceptionally for EncToKs, use cipher instance instead of
            # the decrypt function, due to the difference of structure.
            # Consider updating this
            cipher = aes.AES(self.initial_key, mode="ECB", iv=None)  # create instance of AES class
            for x in self.EncToKs:  # iterate on ToKs to encrypt them one by one
                i = self.EncToKs.index(x)
                decrypted_value = cipher.decrypt(self.EncToKs[i][2], self.EncToKs[i][1])
                self.DecToKs[i] = [i, decrypted_value, False]

            # reply to server with the 'Ready' message
            self.response_subject = "Ready"
            self.response_content = "Ready"
            self.send()
            return True

        # if the message is entitled as 'Model', the training shall start
        elif self.message_subject == "Model":

            # check if the model received is an initial or aggregated one

            # if aggregated
            if self.message_content["stage"] == "aggregated":
                # => extract the parameters from message
                parameters = self.message_content["parameters"]
                # => decrypt parameters with HE
                # To be implemented
                # => load parameters to the available model
                model_class = local_model.local_model(self.local_model)
                self.local_model = deepcopy(model_class.set_parameters(parameters))
            # if initial
            elif self.message_content["stage"] == "initial":
                self.local_model = deepcopy(self.message_content["model"])
                model_class = local_model.local_model(self.local_model)

            # inform the user that local training was started
            print("\n\t/|\\ Started training the model on local data /|\\")

            # train model using the train datasets
            self.local_model.fit(self.x_train, self.y_train)

            # inform the model that the model training is complete
            print("\t/|\\ Finished training the model on local data/|\\\n")

            # predict to evaluate
            predicted = self.local_model.predict(self.x_test)

            model_class.evaluate(predicted, self.y_test)

            # reply to server with the 'model' message
            self.response_subject = "Model"
            self.response_content = model_class.get_parameters()
            self.send()

            return True

        elif self.message_subject == "Hibernate":
            print("\n\t/\\/ Waiting till all clients send their parameters \\/\\")

            return True

        # if the message is entitled as 'Done', the code should be terminated and connection should be close
        elif self.message_subject == "Done":
            self.response_subject = "Disconnect"
            self.response_content = "Disconnect"
            self.send()

            print("\n\t<===> Model training finished. Connection will be terminated")
            self.socket.close()
            # close connection here

            return False

        # if the message carries any other title, this will not be accepted and code should terminate
        else:
            print("\n\tUnrecognized message type: {subject}.".format(subject=self.message_subject))
            return True

    def send(self):
        print("\n\t<==== Sending a \'{subject}\' message to the server".format(subject=self.response_subject))
        self.response = {"subject": self.response_subject, "content": self.response_content}
        # print the message sent if the print option is activated for sent messages
        if self.print_sent_messages:
            print("\n\tMessage to be sent: ", self.response)

        if self.response_subject == "Connect":
            message = pickle.dumps(self.response)
            self.socket.sendall(message)
        else:
            encrypted_response = self.encrypt(pickle.dumps(self.response))
            if self.print_sent_messages:
                print("\tEncrypted version of the message", encrypted_response)
            self.socket.sendall(pickle.dumps(encrypted_response))

    def encrypt(self, message):
        index, key = self.selectEncKey()
        cipher = aes.AES(key=key, mode="ECB", iv=None)
        encrypted_message = cipher.encrypt(message)
        return {"Index": index, "Message": encrypted_message}

    def selectEncKey(self):
        # reset all flags to false when all keys have been used
        for key in self.DecToKs: key.__setitem__(2, False) if all(key[2] for key in self.DecToKs) else None
        # select a random index to choose the key based on it
        i = random.randint(0, (len(self.DecToKs) - 1))
        if self.DecToKs[i][2]:
            return self.selectEncKey()
        else:
            self.DecToKs[i][2] = True
            index = self.DecToKs[i][0]
            key = self.DecToKs[i][1]
            return index, key

    def receive(self):
        """
        this function works on receiving messages from server based on the flow described above
        :return: the data received into the message and the message status
        """
        self.message_buffer = b""
        while True:
            """
                to ensure more than one message is exchanged using the same socket, this while loop is used.
                In this loop, the client receive a message from server and aggregate it and send its content
                and the message status to the run function to act accordingly 
            """
            try:
                # flush all variables to reset the send/receive cycle
                self.all_data_received_flag = False
                # self.message_buffer = b""
                self.temp_buffer = b""

                # recursive call for the receiving function
                self.socket.settimeout(self.receive_timeout)
                self.temp_buffer = self.socket.recv(self.buffer_size)
                self.message_buffer += self.temp_buffer
                try:
                    # if the below code line is passed, then all data have been received
                    self.received_message = pickle.loads(self.message_buffer)
                    self.all_data_received_flag = True
                except BaseException as e:
                    # print("\n\tAn error occurred while reading the received message: ", e)
                    pass

                if self.temp_buffer == b"":
                    # In this case, nothing has been received from client
                    self.message_buffer = b""
                    # when nothing is received by the server for the specified timeout, a 0 status is returned so
                    # the connection will be closed with the client
                    if (time.time() - self.receive_timeout) > self.receive_timeout:
                        return None, 0

                elif self.all_data_received_flag:
                    print("\n\t====> You received a new message from server of size ({data_len} bytes)"
                          .format(data_len=len(self.message_buffer)))

                    # only if the obtained data is not None, return it with a 1 status to the reply function
                    if len(self.message_buffer) > 0:
                        try:
                            self.received_message = pickle.loads(self.message_buffer)
                            return self.received_message, 1
                        except BaseException as e:
                            print("\n\tError Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0
                else:
                    self.receive_start_time = time.time()

            except BaseException as e:
                print("\n\tError while receiving data from the server: {msg}.".format(msg=e))
                return None, 0

    def decrypt(self, key, message):
        cipher = aes.AES(key, mode="ECB", iv=None)
        return pickle.loads(cipher.decrypt(message["Message"][1], message["Message"][0]))
