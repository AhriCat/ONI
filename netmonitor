import psutil
import time

class NetworkMonitor:
    def __init__(self, interval=1):
        """
        Initialize the network monitor.
        
        Args:
            interval (int): Time interval (in seconds) between network usage updates.
        """
        self.interval = interval
        self.last_received = 0
        self.last_sent = 0
        self.running = False

    def start(self):
        """
        Start monitoring network usage.
        """
        self.running = True
        self.last_received, self.last_sent = self.get_current_stats()
        print(f"Monitoring started. Interval: {self.interval} second(s).\n")
        while self.running:
            time.sleep(self.interval)
            self.display_network_usage()

    def stop(self):
        """
        Stop monitoring network usage.
        """
        self.running = False
        print("\nMonitoring stopped.")

    def get_current_stats(self):
        """
        Get the current network statistics (bytes sent and received).
        
        Returns:
            tuple: (bytes_received, bytes_sent)
        """
        net_io = psutil.net_io_counters()
        return net_io.bytes_recv, net_io.bytes_sent

    def display_network_usage(self):
        """
        Calculate and display the current network usage.
        """
        current_received, current_sent = self.get_current_stats()
        received_diff = current_received - self.last_received
        sent_diff = current_sent - self.last_sent

        print(f"Received: {self.convert_bytes(received_diff)} | Sent: {self.convert_bytes(sent_diff)}")
        
        # Update for the next interval
        self.last_received = current_received
        self.last_sent = current_sent

    def convert_bytes(self, num):
        """
        Convert bytes to a human-readable format (KB, MB, etc.).
        
        Args:
            num (int): Number of bytes.
        
        Returns:
            str: Human-readable string representation of bytes.
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if num < 1024.0:
                return f"{num:.2f} {unit}"
            num /= 1024.0
netmon = NetworkMonitor
