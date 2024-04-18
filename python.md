

# Python Cheatsheet

## len()
```python
#note not length, (dont bully me I have been using alot of js...)
len(n)

```

## Array

```python
>>> a = [66.25, 333, 333, 1, 1234.5]
>>> print a.count(333), a.count(66.25), a.count('x')
2 1 0
>>> a.insert(2, -1)
>>> a.append(333)
>>> a
[66.25, 333, -1, 333, 1, 1234.5, 333]
>>> a.index(333)
1
>>> a.remove(333)
>>> a
[66.25, -1, 333, 1, 1234.5, 333]
>>> a.reverse()
>>> a
[333, 1234.5, 1, 333, -1, 66.25]
>>> a.sort()
>>> a
[-1, 1, 66.25, 333, 333, 1234.5]
>>> a.pop()
1234.5
>>> a
[-1, 1, 66.25, 333, 333]

```

## Dict

```python
# Basic Dict
>>> my_cat = {
...  'size': 'fat',
...  'color': 'gray',
...  'disposition': 'loud',
... }
>>> my_cat['age_years'] = 2
>>> print(my_cat)
...
# {'size': 'fat', 'color': 'gray', 'disposition': 'loud', 'age_years': 2}


>>> pet = {'color': 'red', 'age': 42}
>>> for key, value in pet.items():
...     print(f'Key: {key} Value: {value}')
...
# Key: color Value: red
# Key: age Value: 42




```

## Dict.get()

```python
#dict.get()
#The get() method returns the value of an item with the given key. If the key doesn’t exist, it returns None:
>> wife = {'name': 'Rose', 'age': 33}

>>> f'My wife name is {wife.get("name")}'
# 'My wife name is Rose'

>>> f'She is {wife.get("age")} years old.'
# 'She is 33 years old.'

>>> f'She is deeply in love with {wife.get("husband")}'
# 'She is deeply in love with None'

#You can also change the default None value to one of your choice:
>>> wife = {'name': 'Rose', 'age': 33}

>>> f'She is deeply in love with {wife.get("husband", "lover")}'
# 'She is deeply in love with lover'
```

## set

```python
# Create a set from an item 

nums = [1, 2, 2,1,3]
n = set(nums)


>>> s = {1, 2, 3}
>>> s.discard(3)
>>> s
# {1, 2}
>>> s.discard(3)# Using the add() method we can add a single element to the set.
>>> s = {1, 2, 3}
>>> s.add(4)
>>> s
# {1, 2, 3, 4}

# And with update(), multiple ones:

>>> s = {1, 2, 3}
>>> s.update([2, 3, 4, 5, 6])
>>> s
# {1, 2, 3, 4, 5, 6}

#Both methods will remove an element from the set, but remove() will raise a key error if the value doesn’t exist.

>>> s = {1, 2, 3}
>>> s.remove(3)
>>> s
# {1, 2}

>>> s.remove(3)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# KeyError: 3

```

## Set union(), intersection() , difference()

```python
# union() or | will create a new set with all the elements from the sets provided.

>>> s1 = {1, 2, 3}
>>> s2 = {3, 4, 5}
>>> s1.union(s2)  # or 's1 | s2'
# {1, 2, 3, 4, 5}

# intersection or & will return a set with only the elements that are common to all of them.
>>> s1 = {1, 2, 3}
>>> s2 = {2, 3, 4}
>>> s3 = {3, 4, 5}
>>> s1.intersection(s2, s3)  # or 's1 & s2 & s3'
# {3}

# difference or - will return only the elements that are unique to the first set (invoked set).
>>> s1 = {1, 2, 3}
>>> s2 = {2, 3, 4}

>>> s1.difference(s2)  # or 's1 - s2'
# {1}

>>> s2.difference(s1) # or 's2 - s1'
# {4}

# symetric_difference or ^ will return all the elements that are not common between them.
>>> s1 = {1, 2, 3}
>>> s2 = {2, 3, 4}
>>> s1.symmetric_difference(s2)  # or 's1 ^ s2'
# {1, 4}



```

## Queue

```python
from collections import deque
queue = deque()
queue.append(1)
queue.append(2)
queue.popleft()
# 1
queue.popleft()
# 2
```

## OOP

```python
class Animal:

  def __init__(self, voice):
    self.voice = voice

# When a class instance is created, the instance variable
# 'voice' is created and set to the input value.

cat = Animal('Meow')

print(cat.voice) # Output: Meow

dog = Animal('Woof') 

print(dog.voice) # Output: Woof
```

## List Comprehension

```python
>>> new_list = [n for n in names if n.startswith('C')]
>>> print(new_list)
# ['Charles', 'Carol']

>>> new_list = [n for n in names if n.startswith('C')]
>>> print(new_list)
# ['Charles', 'Carol']

>>> nums = [1, 2, 3, 4, 5, 6]
>>> new_list = [num*2 if num % 2 == 0 else num for num in nums]
>>> print(new_list)
# [1, 4, 3, 8, 5, 12]
```

## Set Comprehension

```python
>>> b = {"abc", "def"}
>>> {s.upper() for s in b}
{"ABC", "DEF"}
```

## Dict Comprehension

```python
>>> c = {'name': 'Pooka', 'age': 5}
>>> {v: k for k, v in c.items()}
{'Pooka': 'name', 5: 'age'}

# A List comprehension can be generated from a dictionary:

>>> c = {'name': 'Pooka', 'age': 5}
>>> ["{}:{}".format(k.upper(), v) for k, v in c.items()]
['NAME:Pooka', 'AGE:5']
```

## F Strings

```python
>>> new_list = [n for n in names if n.startswith('C')]
>>> print(new_list)
# ['Charles', 'Carol']
```

## enumerate

```python
>>> names = ['Corey', 'Chris', 'Dave', 'Travis']
>>> for index, name in enumerate(names, start=1):
...     print(index, name)
...
# 1 Corey
# 2 Chris
# 3 Dave
# 4 Travis
```

## zip

```python
>>> names = ['Peter Parker', 'Clark Kent', 'Wade Wilson', 'Bruce Wayne']
>>>  heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']
>>> for name, hero in zip(names, heroes):
...     print(f'{name} is actually {hero}')
...
# Peter Parker is actually Spiderman
# Clark Kent is actually Superman
# Wade Wilson is actually Deadpool
# Bruce Wayne is actually Batman
```

## sort() vs sorted()

* sort() will sort the list in-place, mutating its indexes and returning None , whereas sorted() will return a new sorted list leaving the original list unchanged.
* both `O(nlog(n))` time but sorted() creates a new list (so more space)

```python
>>> nums = [1, 5, 2, 7, 3, 4, 6]
>>> sorted(nums)
# [1, 2, 3, 4, 5, 6, 7]
>>> nums
# [1, 5, 2, 7, 3, 4, 6]
>>> nums.sort()
>>> nums
# [1, 2, 3, 4, 5, 6, 7]

>>> letters = ['a', 'z', 'A', 'Z']
>>> letters.sort(key=str.lower)
>>> letters
# ['a', 'A', 'z', 'Z']

```

## Reading and Writing Files

```python
>>> with open('test.txt', 'r') as f:
...     file_contents = f.read()

# Might need to strip file_contents= [line.strip() for line in file_contents]
...
>>> file_contents
# 'Test file'
# 'r' means read only
>>> with open('test.txt', 'r') as f:
...     file_contents = f.readlines()
...
# 'w' means write only
>>> with open('test.txt', 'w') as f:
...     f.write('Test')
...
#'a' means append only
>>> with open('test.txt', 'a') as f:
...     f.write('Test')
...

import os

lines = []
with open('/home/admin/access.log', 'r') as f:
    lines = f.readlines()

for l in lines:
    # print(l)
    line_array  = l.split()
    ip = line_array[0]
    print(ip)

```

## JSON

```python
>>> import json
>>> person_string = '{"name": "Bob", "languages": ["English", "Fench"]}'
>>> person_dict = json.loads(person_string)
>>> for key, value in person_dict.items():
...     print(key, ':', value)
...
# name : Bob
# languages : ['English', 'Fench']
>>> person_dict['languages']
# ['English', 'Fench']
>>> car = {'make': 'Ford', 'model': 'Mustang'}
>>> car_string = json.dumps(car)
>>> print(car_string)
# {"make": "Ford", "model": "Mustang"}
```

## .split()

```python
>>> sentence = 'the quick brown fox jumped over the lazy dog'
>>> words = sentence.split()
>>> words
# ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
>>> words[2]
# 'brown'
>>> words[2][::-1]
# 'nworb'
>>> ' '.join(words)
# 'the quick brown fox jumped over the lazy dog'
>>> ' '.join(reversed(words))
# 'dog lazy the over jumped fox brown quick the'
>>> ' '.join(reversed(sentence.split()))
# 'dog lazy the over jumped fox brown quick the'
```

## .join()

```python
>>> ''.join(['My', 'name', 'is', 'Simon'])
'MynameisSimon'

>>> ', '.join(['cats', 'rats', 'bats'])
# 'cats, rats, bats'

>>> ' '.join(['My', 'name', 'is', 'Simon'])
# 'My name is Simon'

>>> 'ABC'.join(['My', 'name', 'is', 'Simon'])
# 'MyABCnameABCisABCSimon'


```

## .strip()

```python
>>> spam = '    Hello World     '
>>> spam.strip()
# 'Hello World'

>>> spam.lstrip()
# 'Hello World     '

>>> spam.rstrip()
# '    Hello World'

>>> spam = 'SpamSpamBaconSpamEggsSpamSpam'
>>> spam.strip('ampS')
# 'BaconSpamEggs'
```

## Switch Case (python 3.10+)

```python
match term:
    case pattern-1:
         action-1
    case pattern-2:
         action-2
    case pattern-3:
         action-3
    case _:
        action-default
```

## Math Operations

```python

**	Exponent	2 ** 3 = 8
%	Modulus/Remainder	22 % 8 = 6
//	Integer division	22 // 8 = 2
/	Division	22 / 8 = 2.75
*	Multiplication	3 * 3 = 9
-	Subtraction	5 - 2 = 3
+	Addition	2 + 2 = 4
```

## Walrus Operator (python 3.8+)

```python
>>> a = 'hellooooooooooooo'
>>> if ((n := len(a)) > 10):
...     print(f"List is too long ({n} elements, expected <= 10)")

>>> print(my_var:="Hello World!")
# 'Hello world!'

>>> my_var="Yes"
>>> print(my_var)
# 'Yes'

>>> print(my_var:="Hello")
# 'Hello'
...
```

## Built-in Functions

https://docs.python.org/3/library/functions.html#built-in-functions


## JWT

```python 

import jwt 


```


## requests

```python

import requests
from requests.exceptions import HTTPError

URLS = ["https://api.github.com", "https://api.github.com/invalid"]

for url in URLS:
    try:
        response = requests.get(url)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        print("Success!")



# content 

>>> import requests

>>> response = requests.get("https://api.github.com")
>>> response.content
b'{"current_user_url":"https://api.github.com/user", ...}'

>>> type(response.content)
<class 'bytes'>

>>> response.json()
{'current_user_url': 'https://api.github.com/user', ...}
```


## Flask
```python
from flask import Flask, redirect, url_for, request
app = Flask(__name__)


@app.route('/success/<name>')
def success(name):
	return 'welcome %s' % name


@app.route('/login', methods=['POST', 'GET'])
def login():
	if request.method == 'POST':
		user = request.form['nm']
		return redirect(url_for('success', name=user))
	else:
		user = request.args.get('nm')
		return redirect(url_for('success', name=user))


if __name__ == '__main__':
	app.run(debug=True)


```
# Algo

## Big(O)

```

```

Time complexity and space complexity are the two things you generally have to concern yourself with in SRE interviews. Additionally, there is best, average and worst case of time complexity, whereas with space complexity only the worse case is generally of concern.

Interviewers will generally ask you “fastest” and “slowest” algorithms for certain tasks. For instance, for “Array Sorting Algorithms” Selection sort has the worst performance compared to the other algorithms with a time complexity of O(n^2).

In Big-O notation, the order of execution speed from fastest to slowest is:

- O(1)
- O(log n)
- O(n)
- O(n log n)
- O(n^2)
- O(n!)


## Search

### Binary Search

`O(log n)`

- To be used wen array is sorted
- 

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        L, R = 0, len(nums)-1 
        while L <= R:
            m = L + ((R - L) // 2)
            if nums[m] > target:
                R = m - 1 
            elif nums[m] < target:
                L = m + 1
            else:
                return m
        return -1
```

## Sort 

### Insertion Sort 
`O(n^2)`  -- best case O(n)

```python

def insertionSort(arr):
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
        j = i - 1
        while j >= 0 and arr[j + 1] < arr[j]:
            # arr[j] and arr[j + 1] are out of order so swap them 
            tmp = arr[j + 1]
            arr[j + 1] = arr[j]
            arr[j] = tmp
            j -= 1
    return arr
```

### Merge Sert 

* Divided & Concor (can use recursion)
* `log2n `

[From neetcode Mergesort](https://neetcode.io/courses/dsa-for-beginners/11)
```python
# Implementation of MergeSort
def mergeSort(arr, s, e):
    if e - s + 1 <= 1:
        return arr

    # The middle index of the array
    m = (s + e) // 2

    # Sort the left half
    mergeSort(arr, s, m)

    # Sort the right half
    mergeSort(arr, m + 1, e)

    # Merge sorted halfs
    merge(arr, s, m, e)
    
    return arr

# Merge in-place
def merge(arr, s, m, e):
    # Copy the sorted left & right halfs to temp arrays
    L = arr[s: m + 1]
    R = arr[m + 1: e + 1]

    i = 0 # index for L
    j = 0 # index for R
    k = s # index for arr

    # Merge the two sorted halfs into the original array
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # One of the halfs will have elements remaining
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1
```

### Quick Sort

- can happen in place
- can be recserive 
- not stable 
- avg O(nlogn) (worst O(n2))
```

TODO 


```

### Bucket Sort 
- really good but must a small ish range 
- really simple 
- unstable
- o(n)

```python

def bucketSort(arr):
    # Assuming arr only contains 0, 1 or 2
    counts = [0, 0, 0]

    # Count the quantity of each val in arr
    for n in arr:
        counts[n] += 1
    
    # Fill each bucket in the original array
    i = 0
    for n in range(len(counts)):
        for j in range(counts[n]):
            arr[i] = n
            i += 1
    return arr
```




## Linked List 


TODO add Graphic about linked list vs Array

### Single


### Double


### Reverse Linked List 


```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev, curr = None, head

        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev
        ```
## Queues


- FIFO 
```python
def enqueue(self, val):
    newNode = ListNode(val)

    # Queue is non-empty
    if self.right:
        self.right.next = newNode
        self.right = self.right.next
    # Queue is empty
    else:
        self.left = self.right = newNode


def dequeue(self):
    # Queue is empty
    if not self.left:
        return None
    
    # Remove left node and return value
    val = self.left.val
    self.left = self.left.next
    if not self.left:
        self.right = None
    return val

##

```


## Trees 

### Binary Tree

```python
# Definition for a binary tree node.
 class TreeNode:
	 def __init__(self, val=0, left=None, right=None):
		self.val = val
		self.left = left
		self.right = right
```
### Invert Binary Tree

```python
class Solution:

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:

        if not root:
            return None

        tmp = root.left
        root.left = root.right
        root.right = tmp

  
        self.invertTree(root.right)
        self.invertTree(root.left)

        return root
```




## Kadane's Algo

- o(n)


## Trie

Insert Word: O(1)  
Search Word: O(1)  
Search Prefix: O(1)


```python
class TrieNode:
    def __init__(self):
        self.childs = {}
        self.end = False
        
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr.childs:
                curr.childs[c] = TrieNode()
            curr = curr.childs[c]
        curr.end = True

  

    def search(self, word: str) -> bool:
        curr = self.root
        for c in word:
            if c not in curr.childs:
                return False
            curr = curr.childs[c]
        return curr.end

  

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for c in prefix:
            if c not in curr.childs:
                return False
            curr = curr.childs[c]
        return True


# Your Trie object will be instantiated and called as such:

# obj = Trie()

# obj.insert(word)

# param_2 = obj.search(word)

# param_3 = obj.startsWith(prefix)
```
