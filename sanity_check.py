class Dog():
    def __init__(self) -> None:
        self.x = 5

    def add(self,val:float) -> float:
        x = self.x
        return x + val
    

dog = Dog()

dog.add(5)
print(dog.x)