from abc import abstractmethod

class Base:
    def __init__(
        self, 
        n: int,
        x: int
    ):
        self.result = n**x
    
    @abstractmethod
    def print_out(self):
        pass

class Addon(Base):
    def __init__(
        self
    ):
        super().__init__(
            n=2,
            x=4
        )
    
    def print_out(self):
        print(f"{self.result}")

class Second:
    def __init__(self):
        self.add = Addon()
    
if __name__=="__main__":
    cls = Addon()
    cls.print_out()

