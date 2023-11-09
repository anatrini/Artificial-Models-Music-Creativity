########### Fundamentals of Python ###########

# 1 - Hello world
print("Hello, World!")


# 2 - User and input variables
name = input("Enter your name: ")
print("Hello, " + name + "!")


# 3 - Basic Arithmetic Operations
num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))
sum_result = num1 + num2
difference = num1 - num2
product = num1 * num2
quotient = num1 / num2
print("Sum:", sum_result)
print("Difference:", difference)
print("Product:", product)
print("Quotient:", quotient)


# 4 - Conditional Statements
number = float(input("Enter a number: "))
if number > 0:
    print("Positive")
elif number < 0:
    print("Negative")
else:
    print("Zero")


# 5 - Mean Calculation
numbers = input("Enter a series of numbers separated by spaces: ")
numbers_list = [float(num) for num in numbers.split()]
mean = sum(numbers_list) / len(numbers_list)
print("Mean of the entered numbers is:", mean)


# 6 - BMI Calculation 
weight = float(input("Enter your weight (kg): "))
height = float(input("Enter your height (m): "))
bmi = weight / (height ** 2)
print(f"Your BMI is {bmi:.2f}")


# 7 - List comprehension
n = int(input("Enter a positive integer: "))
even_numbers = [x for x in range(1, n+1) if x % 2 == 0]
print("Even numbers up to", n, "are:", even_numbers)


# 8 - Dictionaries
contacts = {"Alice": "123-456-7890", "Bob": "987-654-3210", "Charlie": "555-123-4567"}
name = input("Enter a name: ")
if name in contacts:
    print(f"The phone number for {name} is {contacts[name]}.")
else:
    print(f"{name} is not in the contacts.")


# 9 - File reading
file_path = "test.txt"  # Replace with the path to your text file
with open(file_path, "r") as file:
    content = file.read()
print("File content:")
print(content)


# 10 - Square Root Calculation
def calculate_square_root(number, iterations=100):
    guess = number / 2
    for _ in range(iterations):
        guess = (guess + number / guess) / 2
    return guess

number = float(input("Enter a number: "))
square_root = calculate_square_root(number)
print(f"The square root of {number} is approximately {square_root:.5f}")


# 11 - String manipulation
sentence = input("Enter a sentence: ")
reversed_sentence = sentence[::-1]
print("Reversed sentence:", reversed_sentence)


# 12 - Loops and patterns
rows = int(input("Enter the number of rows: "))
for i in range(1, rows + 1):
    for j in range(1, i + 1):
        print(j, end=" ")
    print()


# 13 - Random number guessing game
import random

lower_limit = 1
upper_limit = 100
random_number = random.randint(lower_limit, upper_limit)

attempts = 0
while True:
    guess = int(input(f"Guess the number between {lower_limit} and {upper_limit}: "))
    attempts += 1
    if guess < random_number:
        print("Too low! Try again.")
    elif guess > random_number:
        print("Too high! Try again.")
    else:
        print(f"Congratulations! You guessed the number {random_number} in {attempts} attempts.")
        break


# 14 - Prime number checker
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

number = int(input("Enter a number to check for prime: "))
if is_prime(number):
    print(f"{number} is a prime number.")
else:
    print(f"{number} is not a prime number.")


# 15 - Fibonacci sequence
def generate_fibonacci(n):
    fibonacci_sequence = [0, 1]
    while len(fibonacci_sequence) < n:
        next_term = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_term)
    return fibonacci_sequence

n_terms = int(input("Enter the number of Fibonacci terms to generate: "))
fibonacci = generate_fibonacci(n_terms)
print("Fibonacci Sequence:", fibonacci)