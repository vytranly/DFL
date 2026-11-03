"""
P2P Communication layer — TCP socket.
Mỗi node lắng nghe tại localhost:BASE_PORT+node_id.
Giao thức: [4 bytes độ dài] + [N bytes pickle payload]
"""
import socket, pickle, struct, time
from collections import OrderedDict

HEADER_SIZE = 4
RECV_CHUNK  = 4096


def _send_all(sock, data: bytes):
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def _recv_all(sock) -> bytes:
    raw_len = b""
    while len(raw_len) < HEADER_SIZE:
        chunk = sock.recv(HEADER_SIZE - len(raw_len))
        if not chunk:
            raise ConnectionError("Socket closed before header")
        raw_len += chunk
    msg_len = struct.unpack(">I", raw_len)[0]
    data = b""
    while len(data) < msg_len:
        chunk = sock.recv(min(RECV_CHUNK, msg_len - len(data)))
        if not chunk:
            raise ConnectionError("Socket closed mid-message")
        data += chunk
    return data


class NodeServer:
    """Lắng nghe kết nối từ các node khác."""
    def __init__(self, node_id: int, base_port: int):
        self.node_id = node_id
        self.port    = base_port + node_id
        self.sock    = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.listen(10)
        self.sock.settimeout(60.0)

    def receive_one(self):
        """Nhận 1 message. Trả về (sender_id, weights, num_samples, bytes_recv)."""
        conn, _ = self.sock.accept()
        conn.settimeout(30.0)
        try:
            raw     = _recv_all(conn)
            payload = pickle.loads(raw)
            return (payload["sender_id"],
                    payload["weights"],
                    payload["num_samples"],
                    len(raw))
        finally:
            conn.close()

    def close(self):
        self.sock.close()


class NodeClient:
    """Gửi model tới 1 node cụ thể."""
    @staticmethod
    def send(target_port: int, sender_id: int,
             weights: OrderedDict, num_samples: int,
             retries: int = 5, delay: float = 1.0) -> int:
        payload = pickle.dumps({
            "sender_id"  : sender_id,
            "weights"    : weights,
            "num_samples": num_samples,
        })
        for attempt in range(retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect(("127.0.0.1", target_port))
                _send_all(sock, payload)
                sock.close()
                return len(payload)
            except (ConnectionRefusedError, OSError):
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise
        return 0
