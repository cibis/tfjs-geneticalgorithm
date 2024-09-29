FROM node

# Install tfjs-node
RUN yarn add seedrandom@^3.0.5
RUN yarn add @tensorflow/tfjs
RUN yarn add @tensorflow/tfjs-node

RUN yarn add amqplib

COPY "worker.js" .


EXPOSE 3000
CMD [ "node", "worker.js"]
