# infra/zmq_config.py
"""
ZeroMQ socket configuration and constants.

Provides socket factories for PUB/SUB pattern used in the
distributed price pipeline.
"""

import zmq
import config


def create_pub_socket(context: zmq.Context = None) -> zmq.Socket:
    """
    Create a ZeroMQ PUB socket bound to the configured address.

    Parameters
    ----------
    context : zmq.Context, optional
        ZMQ context. Creates a new one if not provided.

    Returns
    -------
    zmq.Socket
        Bound PUB socket ready to send messages.
    """
    if context is None:
        context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(config.ZMQ_PUB_ADDRESS)
    return socket


def create_sub_socket(topics: list[str] = None,
                      context: zmq.Context = None) -> zmq.Socket:
    """
    Create a ZeroMQ SUB socket connected to the configured address.

    Parameters
    ----------
    topics : list[str], optional
        List of topic filters to subscribe to. Empty list or None subscribes
        to all messages.
    context : zmq.Context, optional
        ZMQ context. Creates a new one if not provided.

    Returns
    -------
    zmq.Socket
        Connected SUB socket ready to receive messages.
    """
    if context is None:
        context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(config.ZMQ_SUB_ADDRESS)

    if topics:
        for topic in topics:
            socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    else:
        socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all

    return socket


# Message format constants
TOPIC_PRICE = "PRICE"
TOPIC_SIGNAL = "SIGNAL"
TOPIC_CONTROL = "CTRL"
DELIMITER = "|"
