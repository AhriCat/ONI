import socket
import threading

class PortScanner:
    def __init__(self, target, start_port, end_port):
        self.target = target
        self.start_port = start_port
        self.end_port = end_port

    def scan_port(self, port):
        scanner = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        scanner.settimeout(1)  # Timeout for the connection attempt
        try:
            scanner.connect((self.target, port))
            print(f'[+] Port {port} is open')
        except:
            pass  # Ignore closed ports
        finally:
            scanner.close()

    def start_scan(self):
        print(f'Starting scan on {self.target} from port {self.start_port} to {self.end_port}...')
        for port in range(self.start_port, self.end_port + 1):
            thread = threading.Thread(target=self.scan_port, args=(port,))
            thread.start()


    def run():
        target = input("Enter the target IP address: ")
        start_port = int(input("Enter the starting port: "))
        end_port = int(input("Enter the ending port: "))

        scanner = PortScanner(target, start_port, end_port)
        scanner.start_scan()

ps = PortScanner
