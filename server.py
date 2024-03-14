import hashlib
import random
import socket
import threading
import pickle
import time
from copy import deepcopy
import aes
import global_model
from poly import Poly

clients_list = []
clients_models_parameters = []


class server:
    """
    This class handles starting the hosting server and receiving the incoming messages from clients
    """

    def __init__(self, host, port, backlog_queue, minimum_clients, timeout, buffer_size, table_size,
                 print_incoming_messages, print_sent_messages, print_model_summary, print_model_performance, model,
                 aggregation_method, x_test, y_test, global_rounds):
        """
        class constructor
        :param host: IP of server
        :param port: port of application
        :param backlog_queue: size of listening queue
        """
        self.aggregation_method = aggregation_method
        self.print_sent_messages = print_sent_messages
        self.print_incoming_messages = print_incoming_messages
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.backlogQueue = backlog_queue
        self.minimum_clients = minimum_clients
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.current_rounds = 1
        self.global_rounds = global_rounds
        self.y_test = y_test
        self.x_test = x_test
        self.table_size = table_size
        self.model = model
        self.print_model_summary = print_model_summary
        self.print_model_performance = print_model_performance

    def start(self):
        """
        this function creates the socket that will handle the connection with the server
        :return: None
        """
        # create and bind socket to initiate listening
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(self.backlogQueue)

        # announce the connection to log
        print(f"\nConnection started, Server is listening on {self.host}:{self.port}")

        # initiate listening thread
        client_thread = listen_thread(server_socket=server_socket, minimum_clients=self.minimum_clients,
                                      buffer_size=self.buffer_size, timeout=self.timeout,
                                      print_incoming_messages=self.print_incoming_messages,
                                      print_sent_messages=self.print_sent_messages,
                                      print_model_summary=self.print_model_summary,
                                      print_model_performance=self.print_model_performance,
                                      aggregation_method=self.aggregation_method, model=self.model, x_test=self.x_test,
                                      y_test=self.y_test, global_rounds=self.global_rounds,
                                      current_rounds=self.current_rounds, table_size=self.table_size)
        client_thread.start()


class listen_thread(threading.Thread):
    """
    this class handles the listening into a thread to ensure exchanging more than a message using same socket
    """

    def __init__(self, server_socket, minimum_clients, buffer_size, timeout, print_incoming_messages,
                 print_sent_messages, print_model_summary, print_model_performance, aggregation_method, model,
                 x_test, y_test, global_rounds, current_rounds, table_size):
        """
        class constructor
        :param server_socket: socket details
        :param EncToKs: Encrypted Table of Encryption Keys
        """
        threading.Thread.__init__(self)
        self.current_rounds = current_rounds
        self.global_rounds = global_rounds
        self.y_test = y_test
        self.x_test = x_test
        self.aggregation_method = aggregation_method
        self.print_sent_messages = print_sent_messages
        self.print_incoming_messages = print_incoming_messages
        self.socket = server_socket
        self.minimum_clients = minimum_clients
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.table_size = table_size
        self.model = model
        self.print_model_summary = print_model_summary
        self.print_model_performance = print_model_performance

    def run(self):
        """
        this function runs the listening thread
        :return: None
        """

        while True:
            try:
                """
                In this while loop, the clients connections are accepted and threads created to handle exchanging
                more than one message in the same socket
                """
                client_socket, client_address = self.socket.accept()

                communication_thread = communicate_thread(client_socket=client_socket, client_address=client_address,
                                                          buffer_size=self.buffer_size, timeout=self.timeout,
                                                          receive_start_time=time.time(),
                                                          minimum_clients=self.minimum_clients,
                                                          print_incoming_messages=self.print_incoming_messages,
                                                          print_sent_messages=self.print_sent_messages,
                                                          print_model_summary=self.print_model_summary,
                                                          print_model_performance=self.print_model_performance,
                                                          aggregation_method=self.aggregation_method, model=self.model,
                                                          x_test=self.x_test, y_test=self.y_test,
                                                          global_rounds=self.global_rounds,
                                                          current_rounds=self.current_rounds,
                                                          table_size=self.table_size)
                communication_thread.start()
            except BaseException as e:
                print("\nError while running the communication thread with client: {msg}".format(msg=e))
                break


class communicate_thread(threading.Thread):
    """
    this class handles the communication between server and client. In this class, the send/receive logic is
    implemented following the below logic:
        1. receive the connection from client
        2. receive a "Connect" message from client
        3. Send EncToKs
        4. receive the "Ready" message from client
        5. send the "Model" message from client
        6. loop on 5 until the model converges
        7. opt out of the loop
    """

    def __init__(self, client_socket, client_address, buffer_size, timeout, receive_start_time, minimum_clients,
                 print_incoming_messages, print_sent_messages, print_model_summary, print_model_performance,
                 aggregation_method, model, x_test, y_test, global_rounds, current_rounds, table_size):
        """
        class constructor
        :param client_socket: client socket details
        :param client_address: client IP and Port
        :param buffer_size: standard size of message
        :param timeout: time the server should wait before dropping connection with client if no messages exchanged
        :param receive_start_time: time the server received the message and is used to define the timeout
        :param EncToKs: Encrypted Table of Encryption Keys
        """
        threading.Thread.__init__(self)
        self.table_size = table_size
        self.current_round = current_rounds
        self.global_rounds = global_rounds
        self.y_test = y_test
        self.x_test = x_test
        self.aggregation_method = aggregation_method
        self.print_sent_messages = print_sent_messages
        self.print_incoming_messages = print_incoming_messages
        self.lock = threading.Lock()
        self.client_socket = client_socket
        self.client_address = client_address
        self.buffer_size = buffer_size
        self.receive_timeout = timeout
        self.receive_start_time = receive_start_time
        self.minimum_clients = minimum_clients
        self.clients = []
        self.lock = threading.Lock()
        self.clients_num = 0
        self.response_content = None
        self.response_subject = None
        self.message_subject = None
        self.message_content = None
        self.response = None
        self.all_data_received_flag = False
        self.message_buffer = b""
        self.temp_buffer = b""
        self.received_message = None
        self.decrypted_message = None
        self.connection_status = 0
        self.ToKs = None
        self.EncToKs = None
        self.total_bytes = 0
        self.model = model
        self.aggregated_model = None
        self.print_model_summary=print_model_summary
        self.print_model_performance=print_model_performance

    def receive(self):
        """
        this function receives messages from client. Following the previously described logic, the server cheks
        the content of the message and send the content along with the status of the message to the reply function
        :return:    received_data: the content of the message
                    status: status of the message
        """
        self.message_buffer = b""
        while True:
            """
            To ensure all data sent by client is received, this while loop is used. A recursion in this function
            is used to call the receive from socket function 
            """
            try:
                # flush all receive variables to reset the send/receive cycle
                self.received_message = None
                self.message_subject = None
                self.response = None
                self.all_data_received_flag = False
                self.temp_buffer = b""

                # recursive call for the receiving function
                self.temp_buffer = self.client_socket.recv(self.buffer_size)
                self.message_buffer += self.temp_buffer

                try:
                    # if the below code line is passed, then all data have been received
                    pickle.loads(self.message_buffer)
                    self.all_data_received_flag = True
                except BaseException:
                    # print("\n\tAn error occurred while reading the received message: ", e)
                    pass

                if self.temp_buffer == b"":
                    # In this case, nothing has been received from client
                    self.message_buffer = b""
                    # when nothing is received by the server for the specified timeout, a 0 status is returned so
                    # the connection will be closed with the client
                    if (time.time() - self.receive_timeout) > self.receive_timeout:
                        return None, 0

                # if all data has been collected then the server load it, and send it back the reply function
                elif self.all_data_received_flag:
                    print("\n\t====> You received a new message of size ({data_len} bytes) from {client_info}."
                          .format(client_info=self.client_address, data_len=len(self.message_buffer)))

                    # only if the obtained data is not None, return it with a 1 status to the reply function
                    if len(self.message_buffer) > 0:
                        try:
                            self.received_message = pickle.loads(self.message_buffer)
                            self.total_bytes += len(self.message_buffer)
                            return self.received_message, 1
                        except BaseException as e:
                            print("\n\tError Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0
                else:
                    self.receive_start_time = time.time()

            except BaseException as e:
                print("\n\tError Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def reply(self, received_message):
        # flush all reply variables to reset exchange cycle
        self.message_subject = None
        self.response_subject = None
        self.response_content = None
        self.decrypted_message = None
        """
        In this function, the server will act based on the input received from the "receive" function.
        Based on the message received, there are the following scenarios:
            1.  If the received message is not of type 'dictionary' opt out and print an error
            2.  If the received message is of type 'dictionary', then the below will apply:
                a.  if the message does not contain the keys 'subject' and 'content', print an error
                b.  if the message contain those two keys, then the below will apply:
                    i.   if the message's subject is "Connect", reply with the EncToKs
                    ii.  if the message's subject is "Ready", reply with the model
                    iii. if the message's subject is "Model", assess it's quality and apply the below
                        1. if the model's quality is accepted, then send "Done" messages to clients,
                           so they can opt out and close their connections
                        2. if the model's quality is not accepted, then send back the updated model in a
                           "model" message so that it can be enhanced

        :param received_data:
        :return:
        """

        # print the received message if the printing option is activated
        if self.print_incoming_messages:
            print("\n\tThe message received: ", self.received_message)

        # check if the type of the message if 'dictionary'
        if type(received_message) is dict:
            if "Index" in received_message:
                # First we will decrypt the message using the key with the associated index
                self.decrypted_message = self.decrypt(key=self.ToKs[received_message["Index"]][1],
                                                      message=received_message)

                # Read the message subject after decryption
                self.message_subject = self.decrypted_message["subject"]
                self.message_content = self.decrypted_message["content"]

                # print the decrypted received message if the printing option is activated
                if self.print_incoming_messages:
                    print("\tThe decrypted version is: ", self.decrypted_message)

            else:
                self.message_subject = self.received_message["subject"]
                self.message_content = self.received_message["content"]

            if (self.message_subject is not None) and (self.message_content is not None):

                if self.message_subject == "Connect":
                    """
                    when receiving "Connect" message, encapsulate EncToKs with the response  
                    """
                    self.lock.acquire()
                    clients_list.append(self.client_socket)
                    self.lock.release()

                    """
                    when the client connects, the Table of Encryption keys is created, encrypted and
                    sent to the client to use it in the messages exchange. However, this table is encrypted
                    using an initial key that is generated based on the formular provided in generate_initial_key
                    function below. The key is generated based on the formula applied on the content attached in
                    the connect message, which is generated on the client side
                    """
                    # read the shuffled secret from the "Connect" message
                    shuffled_secret = self.message_content[-32:]

                    # rebuild the original secret after being re-arranged
                    """
                    1. slice the string to 4 substrings of 8 characters.
                    2. characters 8 => 16 moved to 0  => 8
                    3. characters 24 => 32 moved to 8 => 16
                    4. characters 0 => 8 are reversed and become 16 => 24
                    5. characters 16 => 24 are reversed and become 24 => 32
                    """
                    random_secret = shuffled_secret[8:16] + shuffled_secret[24:] + \
                                    shuffled_secret[0:8][::-1] + shuffled_secret[16:24][::-1]

                    initial_key = self.generate_initial_key(self.client_address, random_secret)

                    # generate Table of Keys (ToKs) to be used in encryption later
                    poly_enc = Poly(table_size=self.table_size,
                                    global_key_aes=initial_key)  # create instance of poly class
                    self.ToKs = poly_enc.generate_table_of_keys(
                        self.table_size)  # generate the table by calling the function

                    # Encrypt the ToKs with the global key
                    self.EncToKs = [[] for x in range(self.table_size)]  # create the variable
                    cipher = aes.AES(initial_key, mode="ECB", iv=None)  # create instance of AES class
                    for x in self.ToKs:  # iterate on ToKs to encrypt them one by one
                        i = self.ToKs.index(x)
                        ciphered_key = cipher.encrypt(self.ToKs[i][1])
                        self.EncToKs[i] = [i, ciphered_key[0], ciphered_key[1]]

                    self.response_subject = "EncToKs"
                    self.response = {"subject": self.response_subject, "content": self.EncToKs}

                elif self.message_subject == "Ready":
                    """
                    when receiving "Ready" message, encapsulate the model with the response to start training
                    """
                    # Encapsulate the model parameters in the message to be sent to client
                    self.response_subject = "Model"
                    self.response = {"subject": self.response_subject,
                                     "content": {"model": self.model, "stage":"initial"}}
                elif self.message_subject == "Model":
                    """
                        when receiving "Model" message, assess the model received, if its quality is accepted,
                        then terminate, else, reply with another "Model" message
                    """
                    try:
                        client_model_parameters = self.decrypted_message["content"]

                        if client_model_parameters is not None:
                            clients_models_parameters.append(client_model_parameters)
                        else:
                            print("\n\tThe received gradients are not feasible")
                    except BaseException as e:
                        print("\n\tError occurred: {msg}".format(msg=e))

                    if len(clients_models_parameters) >= self.minimum_clients:
                        print("\n\t/|\\ Parameters received from all clients /|\\")

                        global_ = global_model.global_model(clients_models_parameters, self.model, self.x_test, self.y_test,
                                                            self.current_round, self.print_model_summary,
                                                            self.print_model_performance)
                        # aggregate the models
                        self.model, aggregated_parameters = global_.aggregate()
                        # currently, the global model is stored in the self.model variable. we should work on returning
                        # this model to the main class so that the user can save it if needed.

                        self.response_subject = "Model"
                        self.response = {"subject": self.response_subject,
                                         "content": {"parameters": aggregated_parameters, "stage": "aggregated"}}

                        # increment the training rounds counter
                        self.current_round += 1

                        # reset the parameters list to ensure waiting new parameters
                        clients_models_parameters.clear()

                        # If the server trained the global model with the number of rounds defined, then it sends to
                        # all clients a "Done" message, where they disconnect. The server by its turn prompts the user
                        # if they want to save the model
                        if self.current_round > self.global_rounds:
                            print("\\\\|// Finished training the global model \\\\|//\n"
                                  "===============================================================================")
                            self.response_subject = "Done"
                            self.response_content = "Done"
                            self.response = {"subject": self.response_subject, "content": self.response_content}

                            encrypted_response = self.encrypt(pickle.dumps(self.response))
                            if self.print_sent_messages:
                                print("\tEncrypted version of the message: ", encrypted_response)
                            self.client_socket.sendall(pickle.dumps(encrypted_response))

                            for member in clients_list:
                                member.sendall(pickle.dumps(self.response))
                            return True
                        else:
                            for member in clients_list:
                                member.sendall(pickle.dumps(self.response))
                            return True
                    else:
                        self.response_subject = "Hibernate"
                        self.response_content = "Wait until all clients send their parameters"
                        self.response = {"subject": self.response_subject, "content": self.response_content}

                elif self.message_subject == "Disconnect":
                    return False
                else:
                    self.response = pickle.dumps("The message received was not recognized by the server")
                    return True

                try:
                    print("\n\t<==== Sending a \'{subject}\' message to client: ".format(subject=self.response_subject))

                    if self.print_sent_messages:
                        print("\n\tMessage to be sent: ", self.response)

                    if self.response_subject == "EncToKs":
                        self.client_socket.sendall(pickle.dumps(self.response))
                    else:
                        encrypted_response = self.encrypt(pickle.dumps(self.response))
                        if self.print_sent_messages:
                            print("\tEncrypted version of the message: ", encrypted_response)
                        self.client_socket.sendall(pickle.dumps(encrypted_response))

                    return True

                except BaseException as e:
                    print("\n\t<==== Error Sending Data to the Client: {msg}.\n".format(msg=e))
            else:
                print(
                    "\n\tThe received dictionary from the client must have the 'subject' and 'content' keys available")
        else:
            print("\n\tA dictionary is expected to be received from the client but {d_type} received.".format(
                d_type=type(received_message)))

    def run(self):
        """
        this function is responsible for running the communication thread. Threads ensure that the code will
        not be blocked on a specific point
        :return:
        """
        print("\n-----------------------------------------------------------------------\n"
              "The client {client} joined the network. "
              "\n-----------------------------------------------------------------------"
              .format(client=self.client_address))
        # This while loop allows the server to wait for the client to send data more than once within the same
        # connection.
        while True:
            self.receive_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "\n\t<===> Waiting to Receive Data from {client} " \
                        "Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT" \
                .format(client=self.client_address, year=time_struct.tm_year, month=time_struct.tm_mon,
                        day=time_struct.tm_mday, hour=time_struct.tm_hour,
                        minute=time_struct.tm_min, second=time_struct.tm_sec)
            print(date_time)

            self.received_message, self.connection_status = self.receive()

            if self.connection_status == 0:
                self.client_socket.close()
                print("\nConnection Closed with: ", self.client_address,
                      ". Total size of exchanged messages: ", self.total_bytes, " Bytes")
                break

            if not self.reply(self.received_message):
                print("\nConnection Closed with: ", self.client_address,
                      ". Total size of exchanged messages: ", self.total_bytes, " Bytes")
                break

    def encrypt(self, message):
        index, key = self.selectEncKey()
        cipher = aes.AES(key=key, mode="ECB", iv=None)
        encrypted_message = cipher.encrypt(message)
        return {"Index": index, "Message": encrypted_message}

    def decrypt(self, key, message):
        cipher = aes.AES(key, mode="ECB", iv=None)
        return pickle.loads(cipher.decrypt(message["Message"][1], message["Message"][0]))

    def selectEncKey(self):
        # reset all flags to false when all keys have been used
        for key in self.ToKs: key.__setitem__(2, False) if all(key[2] for key in self.ToKs) else None
        # select a random index to choose the key based on it
        i = random.randint(0, (len(self.ToKs) - 1))
        if self.ToKs[i][2]:
            return self.selectEncKey()
        else:
            self.ToKs[i][2] = True
            index = self.ToKs[i][0]
            key = self.ToKs[i][1]
            return index, key

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
