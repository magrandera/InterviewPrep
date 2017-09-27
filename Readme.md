# Algorithms and Data Structures/Interview Prep

## Theory

### Algorithms

##### Terms
- **Precondition**: formula Pre(x_1,...,x_n) about the inputs
- **Postcondition**: formula Post(x_1,...,x_n,r) about in the inputs and outputs
- **Terminates for**: if running A with these inputs finishes in finitely many steps
- **Terminates**: if it terminates whenever Pre(x_1,...,x_n)
- **Partially Correct**: if for all v_1,...,v_n
  - if Pre(v_1,...,v_n) and
  - A terminates for v_1,...,v_n with return value r then
  - Post(v_1,...,V_n)

## Data Structure Basics

### Array

##### Definition:
- Stores data elements based on an sequential, most commonly 0 based, index.
- Based on [tuples](http://en.wikipedia.org/wiki/Tuple) from set theory.
- They are one of the oldest, most commonly used data structures.  
  
##### What you need to know:
- Optimal for indexing; bad at searching, inserting, and deleting (except at the end).
- **Linear arrays**, or one dimensional arrays, are the most basic.
  - Are static in size, meaning that they are declared with a fixed size.
- **Dynamic arrays** are like one dimensional arrays, but have reserved space for additional elements.
  - If a dynamic array is full, it copies it's contents to a larger array.
- **Two dimensional arrays** have x and y indices like a grid or nested arrays.  
  
##### Big O efficiency:
- Indexing:         Linear array: O(1),      Dynamic array: O(1)
- Search:           Linear array: O(n),      Dynamic array: O(n)
- Optimized Search: Linear array: O(log n), Dynamic array: O(log n)
- Insertion:        Linear array: n/a        Dynamic array: O(n)
  

### Linked List
##### Definition: 
- Stores data with **nodes** that point to other nodes.
  - Nodes, at its most basic it has one datum and one reference (another node).
  - A linked list _chains_ nodes together by pointing one node's reference towards another node.  

##### What you need to know:
- Designed to optimize insertion and deletion, slow at indexing and searching.
- **Doubly linked list** has nodes that reference the previous node.
- **Circularly linked list** is simple linked list whose **tail**, the last node, references the **head**, the first node.
- **Stack**, commonly implemented with linked lists but can be made from arrays too.
  - Stacks are **last in, first out** (LIFO) data structures.
  - Made with a linked list by having the head be the only place for insertion and removal.
- **Queues**, too can be implemented with a linked list or an array.
  - Queues are a **first in, first out** (FIFO) data structure.
  - Made with a doubly linked list that only removes from head and adds to tail.  

##### Big O efficiency:
- Indexing:         Linked Lists: O(n)
- Search:           Linked Lists: O(n)
- Optimized Search: Linked Lists: O(n)
- Insertion:        Linked Lists: O(1)  

### Heap
###### Definition:
- consists of either an Array or a Binary Tree
- Heap[Z,<=] is a min heap. All children bigger than the parent
- Heap[Z,>=] is a max heap. All children smaller than the parent

##### Operations on Heaps:
- insert: O(log n)
- extract: O(log n)
- find: O(log n)

### Hash Table or Hash Map
##### Definition: 
- Stores data with key value pairs.
- **Hash functions** accept a key and return an output unique only to that specific key. 
  - This is known as **hashing**, which is the concept that an input and an output have a one-to-one correspondence to map information.
  - Hash functions return a unique address in memory for that data.

##### What you need to know:
- Designed to optimize searching, insertion, and deletion.
- **Hash collisions** are when a hash function returns the same output for two distinct inputs.
  - All hash functions have this problem.
  - This is often accommodated for by having the hash tables be very large.
- Hashes are important for associative arrays and database indexing.

##### Big O efficiency:
- Indexing:         Hash Tables: O(1)
- Search:           Hash Tables: O(1)
- Insertion:        Hash Tables: O(1)

### General Trees 

##### Definition
connected directed graph in which 
- there is exactly one node with in-degree 0
- all other nodes have in-degree 1

#### Binary Tree
##### Definition: 
- Is a tree like data structure where every node has at most two children.
  - There is one left and right child node.

##### Nodes in a Binary Tree
- has at most 2^n nodes at depth n
- at most 2^{h+1} -1 nodes in total
- if it is perfect, it has exactly 2^n nodes at depth n and exactly 2^{h+1} - 1 nodes in total

##### What you need to know:
- Designed to optimize searching and sorting.
- A **degenerate tree** is an unbalanced tree, which if entirely one-sided is a essentially a linked list.
- They are comparably simple to implement than other data structures.
- Used to make **binary search trees**.
  - A binary tree that uses comparable keys to assign which direction a child is.
  - Left child has a key smaller than it's parent node.
  - Right child has a key greater than it's parent node.
  - There can be no duplicate node.
  - Because of the above it is more likely to be used as a data structure than a binary tree.

##### Big O efficiency:
- Indexing:  Binary Search Tree: O(log n)
- Search:    Binary Search Tree: O(log n)
- Insertion: Binary Search Tree: O(log n) 

### General Set

#### Bit Vectors
##### Definition:
- `Array[bool](m)`
- set the values that are contained in the set to 1

##### Functions & Complexity:
- insert & delete: O(1)
- equal, union, inter, diff: O(m)

#### List Sets
##### Definition:
- Set is represented as a List
- All usual functions for List except insert(a) does nothing if a is already contained in the set

##### Functions & Complexity:
- contains, insert, delete: O(n)

### General Graphs
##### Definition:
- is a pair G = (N,E) such that E is a binary relation on N.
- If E is symmetric, G is called undirected, otherwise directed

##### General Functions of a Graph:
- edge(G,i,j): for two nodes i, j, the edge from i to j
- outgoing(G,i,j): for node i, the list of outgoing edges
- incoming(G,i): for node i, the list of incoming edges
- edges(G): an iterator over all edges

#### Adjacency Matrix:
##### Functions & Complexity:
- drawback is size |N|^2
- edge(G,i,j), edges(G) takes O(1)
- outgoing(G,i) and incoming(G,i) take either O(1) or O(m) depending on the implementation

#### Adjacency List
##### Functions & Complexity:
- size is |N| + |E|
- edge(G,i,j) takes O(d) where d is the maximal number of outgoing edges
- outgoing(G,i) takes O(1)
- incoming is inefficient
- edges(g) takes O(m) because we have to concatenate m iterators

#### Redundant Adjacency List
##### Functions & Complexity:
- size is 2(|N| + |E|)
- edge(G,i,j) takes O(d)
- outgoing(G,i) and incoming(G) take O(1)
- edges(g) takes O(m)

#### Algorithms
##### BFS/DFS
- to make DFS just switch Queue to stack
- r number of reachable nodes, e the number of edges, O(r + e)
```scala
fun BFS(G : Graph, start : N) : List[N] =
    reachable : List[N] = Nil
    horizon : Queue[N] = new Queue[N]()
    enqueue(horizon,start)
    while !empty(horizon)
        i = dequeue(horizon)
        reachable = prepend(i, reachable)
        foreach(outgoing(G,i), j -> 
            if !contains(reachable,j) && !contains(horizon,j)
                enqueue(horizon,j)
        )
    reverse(reachable)
```
##### Kruskal
- O(|E|log|E|)
```scala
fun Kruskal(G : Graph) : Set[E] =
    edges = list of edges e in E, sorted by increasing weight
    sot = new Set[E]()
    foreach(edges, e -> if (isSetOfTrees(sot U {e})){insert(sot,e)})
    sot
```

##### Cheapest Path/Dijkstra
- m is |N|
- O((|N| + |E|)log |N|)
```scala
fun Dijkstra(G : Graph, start : N) : Array[List[N]] =
    cheapest = new Array[List[N]](m)
    foreach(N, n -> cheapest[n] = null)
    cheapest[start] = start
    cost = new Array[N](m)
    foreach(N, n -> cost[n] = infinity)
    cost[start] = 0
    rest = new Set[N]()
    foreach(N, n -> insert(rest,n))
    while !empty(rest)
        i = a node i in rest with minimal value of cost[i]
        delete(rest,i)
        foreach(outgoing(G,i), j ->
            c = cost[i] + weight(i,j)
            if c < cost[j]
                cheapest[j] = append(cheapest[i],j)
                cost[j] = c
        )
    cheapest
```

### Algebraic Data Structure
##### Definition:
- one binary relation U x U -> bool. Ex: Graphs, Preorders, orders
- one binary funciton U x U -> U such as in Monoids, groups
- two binary functions U x U -> U

## Search Basics
### Breadth First Search
##### Definition:
- An algorithm that searches a tree (or graph) by searching levels of the tree first, starting at the root.
  - It finds every node on the same level, most often moving left to right. 
  - While doing this it tracks the children nodes of the nodes on the current level.
  - When finished examining a level it moves to the left most node on the next level.
  - The bottom-right most node is evaluated last (the node that is deepest and is farthest right of it's level). 

##### What you need to know:
- Optimal for searching a tree that is wider than it is deep.
- Uses a queue to store information about the tree while it traverses a tree.
  - Because it uses a queue it is more memory intensive than **depth first search**.
  - The queue uses more memory because it needs to stores pointers

###### Algorithm:
```scala
fun BFS[A](n : Tree[A], f : Tree[A] -> unit) =
    needToVisit = new Queue[Tree[A]]()
    enqueue(needToVisit, n)
    while !empty(needToVisit)
        n = dequeue(needToVisit)
        f(n)
        foreach(n.children, x -> enqueue(needToVisit, x))
```
  
##### Big O efficiency:
- Search: Breadth First Search: O(|E|)
- E is number of edges

### Depth First Search
##### Definition:
- An algorithm that searches a tree (or graph) by searching depth of the tree first, starting at the root.
  - It traverses left down a tree until it cannot go further.
  - Once it reaches the end of a branch it traverses back up trying the right child of nodes on that branch, and if possible left from the right children.
  - When finished examining a branch it moves to the node right of the root then tries to go left on all it's children until it reaches the bottom.
  - The right most node is evaluated last (the node that is right of all it's ancestors). 
  
##### What you need to know:
- Optimal for searching a tree that is deeper than it is wide.
- Uses a stack to push nodes onto.
  - Because a stack is LIFO it does not need to keep track of the nodes pointers and is therefore less memory intensive than breadth first search.
  - Once it cannot go further left it begins evaluating the stack.

###### Algorithm:
```scala
fun DFS[A](n : Tree[A], f : Tree[A] -> unit) =
    f(n)
    foreach(n.children, x -> DFS[A](x, f))
```

##### Big O efficiency:
- Search: Depth First Search: O(|E|)
- E is number of edges

#### Breadth First Search Vs. Depth First Search
- The simple answer to this question is that it depends on the size and shape of the tree.
  - For wide, shallow trees use Breadth First Search
  - For deep, narrow trees use Depth First Search

#### Nuances:
  - Because BFS uses queues to store information about the nodes and its children, it could use more memory than is available on your computer.  (But you probably won't have to worry about this.)
  - If using a DFS on a tree that is very deep you might go unnecessarily deep in the search. See [xkcd](http://xkcd.com/761/) for more information.
  - Breadth First Search tends to be a looping algorithm.
  - Depth First Search tends to be a recursive algorithm.

## Efficient Sorting Basics
### Merge Sort
##### Definition:
- A comparison based sorting algorithm
  - Divides entire dataset into groups of at most two.
  - Compares each number one at a time, moving the smallest number to left of the pair.
  - Once all pairs sorted it then compares left most elements of the two leftmost pairs creating a sorted group of four with the smallest numbers on the left and the largest ones on the right. 
  - This process is repeated until there is only one set.

##### What you need to know:
- This is one of the most basic sorting algorithms.
- Know that it divides all the data into as small possible sets then compares them.

##### Algorithm:
```scala
fun mergesort(x : List[A]) : List[A] =
    n = length(x)
    if n < 2
        x
    else
        k = n div 2
        l = mergesort([0:k-1])
        r = mergesort([k:n-1])
        return merge(l,r)

fun merge(x : List[A], y : List[A]) : List[A] =
    xRest = x
    yRest = y
    res = []
    while nonempty(xrest) || nonempty(yRest)
        takefromX = empty(yRest) || (nonempty(xRest) && xRest.head <= yRest.head)
        if takefromX
            res = cons(xRest.head,res)
            xRest = xRest.tail
        else
            res = cons(yRest.head, res)
            yRest = yRest.tail
    return reverse(res)
```

##### Big O efficiency:
- Best Case Sort: Merge Sort: O(n log n)
- Average Case Sort: Merge Sort: O(n log n)
- Worst Case Sort: Merge Sort: O(n log n)
- Space: O(n)

### Quicksort
##### Definition:
- A comparison based sorting algorithm
  - Divides entire dataset in half by selecting the average element and putting all smaller elements to the left of the average.
  - It repeats this process on the left side until it is comparing only two elements at which point the left side is sorted.
  - When the left side is finished sorting it performs the same operation on the right side.
- Computer architecture favors the quicksort process.

##### What you need to know:
- While it has the same Big O as (or worse in some cases) many other sorting algorithms it is often faster in practice than many other sorting algorithms, such as merge sort.
- Know that it halves the data set by the average continuously until all the information is sorted.

##### Algorithm:
```scala
fun quicksort(x : Array[A]) =
    quicksortSublist(x,0,length(x) - 1)

fun quicksortSublist(x : Array[A], first : N, last : N) =
    if first >= last
        return
    else
        pivot = x[last]
        pivotPos = first
        for j from first to last - 1
            if x[j] <= pivot
                swap(x,pivotPos,j)
                pivotPos += 1
        swap(x,pivotPos,last)
        quicksortSublist(x,first,pivotPos - 1)
        quicksortSublist(x,pivotPos + 1, last)
```

##### Big O efficiency:
- Best Case Sort: O(n log n)
- Average Case Sort: O(n log n)
- Worst Case Sort: O(n^2)
- Space: O(log n)

#### Merge Sort Vs. Quicksort
- Quicksort is likely faster in practice.
- Merge Sort divides the set into the smallest possible groups immediately then reconstructs the incrementally as it sorts the groupings.
- Quicksort continually divides the set by the average, until the set is recursively sorted.

### Bubble Sort
##### Definition:
- A comparison based sorting algorithm
  - It iterates left to right comparing every couplet, moving the smaller element to the left.
  - It repeats this process until it no longer moves and element to the left.

##### What you need to know:
- While it is very simple to implement, it is the least efficient of these three sorting methods.
- Know that it moves one space to the right comparing two elements at a time and moving the smaller on to left.

##### Algorithm:
```scala
fun bubblesort(x : Array[M]) =
    sorted = false
    while !sorted
        sorted = true
        for i from 0 to length(x) - 2
            if !x[i] <= x[i+1]
                sorted = false
                swap(x,i,i+1)
```

##### Big O efficiency:
- Best Case Sort: O(n)
- Average Case Sort: O(n^2)
- Worst Case Sort: O(n^2)
- Space: O(1)

### Insertion Sort
##### Definition:
- Comparison based sorting
  - iterates left to right moving the smaller elements to the front

##### What you need to know:
- Very simple to implement, quite efficient for small arrays 
- mostly more efficient than bubblesort
- in-place

##### Algorithm:
```scala
fun insertionsort(x : Array[A]) =
    for i from 1 to length(x) - 1
        current = x[i]
        pos = i
        while pos > 0 && current < x[pos - 1]
            x[pos] = x[pos-1]
            pos = pos-1
        x[pos] = current
```

##### Big O efficiency:
- Best Case Sort: O(n)
- Average Case Sort: O(n^2)
- Worst Case Sort: O(n^2)
- Space: O(1)

### Heap Sort

##### Definition:
- converts input into a Max Heap
- extracts one element after the other and adds it to the lists

##### What do you need to know:
- in-place 
- non stable sort

##### Algorithm:
```scala
fun heapsort(x : A*) : A* =
    h = new Heap[A,>=]()
    left = x
    for i from 0 to length(x) - 1
        next = left.head
        insert(h,next)
        left = left.tail
    res = Nil
    for i from 0 to length(x) - 1
        next = extract(h)
        res = prepend(next, res)
```

### Big O efficiency:
- Best Case Sort: O(n log n)
- Average Case Sort: O(n log n)
- Worst Case Sort: O(n log n)
- Space: O(1)

## Basic Types of Algorithms

## Algorithms
### Greedy Algorithm
##### Definition:
- An algorithm that, while executing, selects only the information that meets a certain criteria.
- The general five components, taken from [Wikipedia](http://en.wikipedia.org/wiki/Greedy_algorithm#Specifics):
  - A candidate set, from which a solution is created.
  - A selection function, which chooses the best candidate to be added to the solution.
  - A feasibility function, that is used to determine if a candidate can be used to contribute to a solution.
  - An objective function, which assigns a value to a solution, or a partial solution.
  - A solution function, which will indicate when we have discovered a complete solution.

##### What you need to know:
- Used to find the optimal solution for a given problem.
- Generally used on sets of data where only a small proportion of the information evaluated meets the desired result.
- Often a greedy algorithm can help reduce the Big O of an algorithm.

##### General Greedy Algorithm
```scala
fun greedy[M](elements : List[M], Acceptable : Set[M] -> bool) =
    solution = new Set[M]()
    foreach(elements, x -> if (Acceptable(solution U {x})) {insert(solution,x)})
    solution
```

##### Matroid
consists of:
- a finite set M 
- a property Acceptable : Set[M] -> B (subsets with this property are called acceptable)
such that the following holds
- M has at least one acceptable subset.
- Acceptable is subset-closed ie. subsets of acceptable sets are also acceptable
- If A and B are acceptable and |A| > |B|, then B can be increased to an acceptable set B U {x} by adding an element x in A to B.

### Backtracking
##### What you need to know:
- makes sense if:
  - we have little or no information to predict the best choice right away
  - after making a choice, we can detect quickly that it was the wrong choice right away

##### General Algorithm
```scala
fun search(state : List[A]) : Option[List[A]] = 
    if (abort(state)) {return None}
    if (solution(state)) {return Some(state)}
    foreach(choices(state), c -> 
        x = search(state + [c])
        if (x != None) {return x}
    )
    return None
```

### Divide & Conquer
##### Definition:
1. the input is divided into parts
2. each part is processed recursively
3. the results of the parts are combined into the result as of the whole

##### Master Theorem
r = `#`of recursion, d = `#`of subproblems, c = cost of merging
- if r < d^c: O(n^c)
- if r = d^c: O(n^c log_d n)
- if r > d^c: O(n^{log_d r})

### Dynamic Programming
```scala
fun dynamic(n : N) : B = 
    results = new Array[B](n)
    for i in 0,...,n
        r = P(i)
        results[i] = r
    results[n]
```

## Miscellaneous Functions

### Factorial
```scala
fun fact(n : N) : N = 
    product = 1
    factor = 1
    while factor <= n
        product = product * factor
        factor += 1
    return product
```

### GCD
- O(log n)
```scala
fun gcd(m : N, n : N) : N =
    if n == 0
        m
    else
        gcd(n, m mod n) 
```

### Exponentiation
##### Naive
- O(n)
```scala
fun power(x : Z,n : N) : N =
    if n == 0
        1
    else
        x * power(x, n-1)
```

##### Square-and-Multiply
- O(log n)
```scala
fun sqmult(x : Z, n : N) : N =
    if n == 0
        1
    else
        r = sqmult(x, n div 2)
        sq = r * r
        if (n mod 2 == 0) {sq} else{x*sq}
```

### Fibonacci
##### Naive
- O(2^n)
```scala
fun fib(n : N) = 
    if n <= 1
        n
    else
        fib(n-1) + fib(n-2)
```

##### Linear
```scala
fun fib(n : N) : N =
    if n <= 1
        n
    else
        prev = 0
        current = 1
        i = 1
        while i < n
            next = current + prev
            prev = current
            current = next
            i += 1
        return current 
```

##### Inexact
```scala
fun fib(n : N) = round(((1 + sqrt(5))/2)/sqrt(5))
```

##### Sublinear
```scala
(fib(n),fib(n-1)) = (1,0) * {{1,1},{1,0}}^n 
```

## Practical

#### Bit Manipulation
- ~0 creates int of all 1s
- x << y Returns x with the bits shifted to the left by y places (and new bits on the right-hand-side are zeros).
- x >> y Returns x with the bits shifted to the right by y places
- x & y Does a "bitwise and". Each bit of the output is 1 if the corresponding bit of x AND of y is 1, otherwise it's 0.
- x | y Does a "bitwise or". Each bit of the output is 0 if the corresponding bit of x AND of y is 0, otherwise it's 1.
- x ^ y Does a "bitwise exclusive or". Each bit of the output is the same as the corresponding bit in x if that bit in y is 0, and it's the complement of the bit in x if that bit in y is 1.

#### Power of 2
```python
((n & (n-1)) == 0) and n != 0
```

#### All subsets of a list
```python
def subsets(nums):
  result = [[]]
  for x in nums:
    result = result + [y + [x] for y in result]
  return result
```

#### All permuations of a list
```python
def permute(self, l):
  if not l:
    return [[]]
  res = []
  for e in l:
    temp = l[:]
    temp.remove(e)
    res.extend([[e] + r for r in self.permute(temp)])
  return res
```
