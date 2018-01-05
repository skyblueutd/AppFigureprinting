class Packet:
    source = ""
    destination = ""
    def __init__(self, source, destination, length):
        self.src = source
        self.dst = destination
        self.len = length
        self.judge()

    def judge(self):
        if self.src != self.source:
            self.len = 0 - self.len