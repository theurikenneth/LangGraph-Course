# 1. TypeDict
# from typing import TypedDict

# class Movie(TypedDict):
#     name : str
#     year: int

# movie = Movie(name = "Avengers", year = 2019)
# print(movie)

# 2. Union
# from typing import Union

# def square(x: Union[int, float]) -> float:
#     return x * x

# x = 5
# x = 1.234
# x = "I am a string"
# print(square(x))

# 3. Optional
# from typing import Optional

# def nice_message(name: Optional[str]) -> None:
#     if name is None:
#         print("Hey random person")
#     else:
#         print(f"Hi there, {name}")

# name = "Bob"
# nice_message(name)
# nice_message(None)

# 4. Any
# from typing import Any

# def print_value(x: Any):
#     print(x)

# print_value("I pretend to be Batman in the shower sometimes")

# 5. Lambda
square = lambda x: x * x
print(square(10))

nums = [1, 2, 3, 4]
squares = list(map(lambda x: x * x, nums))
print(squares)