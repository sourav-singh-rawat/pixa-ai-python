# class Foo:
#     def abc(self):
#         method = getattr(self, "xyz")
#         method("abc")
    
#     def xyz(self,x:str):
#         print(f"xyz {x}")
        
# if __name__ == "__main__":
#     f = Foo()
#     f.abc()
    

from typing import Callable

class A:
    def __init__(self):
        self.b = B(self.xyz)

    def xyz(self,val) -> None:
        print(val)

class B:
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.execute_callback()

    def execute_callback(self) -> None:
        self.callback('13')

if __name__ == "__main__":
    a = A()