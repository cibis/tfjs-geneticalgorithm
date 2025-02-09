import pika
import os

class RabbitMQ:
    def __init__(self, host, port):
        # self.user = os.getenv('RABBITMQ_USER', 'user')
        # self.password = os.getenv('RABBITMQ_PASSWORD', 'password')
        # self.host = os.getenv('RABBITMQ_HOST', 'localhost')
        # self.port = int(os.getenv('RABBITMQ_PORT', 5672))
        print(f"host: {host}, port: {port}")
        self.host = host
        self.port = int(port)       
        self.connection = None
        self.channel = None
        self.connect()

    def connect(self):
        #credentials = pika.PlainCredentials(self.user, self.password)
        parameters = pika.ConnectionParameters(host=self.host, port=self.port 
                                               #,credentials=credentials
                                               )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()

    def consume(self, queue_name, callback):
        if not self.channel:
            raise Exception("Connection is not established.")
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=False)
        self.channel.start_consuming()
        

    def publish(self, queue_name, message):
        if not self.channel:
            raise Exception("Connection is not established.")
        self.channel.queue_declare(queue=queue_name, durable=True)
        self.channel.basic_publish(exchange='',
                                   routing_key=queue_name,
                                   body=message,
                                   properties=pika.BasicProperties(
                                       delivery_mode=2,  # make message persistent
                                   ))
        print(f"Sent message to queue {queue_name}: {message}")
