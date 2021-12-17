
class A(object):
    def foo1(self):
        self.foo2()
    def foo2(self):
       print ("A.foo2")
class B(A):
    def foo1(self):
        super().foo1()
    def foo2(self):
       print ("B.foo2")

myB = B()
myB.foo1()