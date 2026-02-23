package horosvec

import "sync"

// cachedNode holds graph node data in the LRU cache.
type cachedNode struct {
	nodeID    int32
	neighbors []int32
	vec       []float32 // raw vector for exact L2 during search
	code      []byte    // RaBitQ 1-bit code
	sqNorm    float64
	l1Norm    float64

	// doubly-linked list pointers
	prev, next *cachedNode
}

// nodeCache is a thread-safe LRU cache for graph nodes.
type nodeCache struct {
	mu       sync.RWMutex
	capacity int
	items    map[int32]*cachedNode

	// doubly-linked list: head = most recently used, tail = least recently used
	head, tail *cachedNode
}

// newNodeCache creates an LRU cache with the given capacity.
func newNodeCache(capacity int) *nodeCache {
	return &nodeCache{
		capacity: capacity,
		items:    make(map[int32]*cachedNode, capacity),
	}
}

// get retrieves a node from the cache. Returns nil if not found.
func (c *nodeCache) get(nodeID int32) *cachedNode {
	c.mu.RLock()
	node, ok := c.items[nodeID]
	c.mu.RUnlock()
	if !ok {
		return nil
	}
	// Promote to head (write lock needed)
	c.mu.Lock()
	c.moveToHead(node)
	c.mu.Unlock()
	return node
}

// put adds or updates a node in the cache.
func (c *nodeCache) put(node *cachedNode) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if existing, ok := c.items[node.nodeID]; ok {
		// Update existing
		existing.neighbors = node.neighbors
		existing.vec = node.vec
		existing.code = node.code
		existing.sqNorm = node.sqNorm
		existing.l1Norm = node.l1Norm
		c.moveToHead(existing)
		return
	}

	// Add new
	c.items[node.nodeID] = node
	c.addToHead(node)

	// Evict if over capacity
	for len(c.items) > c.capacity {
		c.evictTail()
	}
}

// clear removes all entries from the cache.
func (c *nodeCache) clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[int32]*cachedNode, c.capacity)
	c.head = nil
	c.tail = nil
}

// size returns the number of cached nodes.
func (c *nodeCache) size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

// --- internal linked list operations (must hold mu) ---

func (c *nodeCache) addToHead(node *cachedNode) {
	node.prev = nil
	node.next = c.head
	if c.head != nil {
		c.head.prev = node
	}
	c.head = node
	if c.tail == nil {
		c.tail = node
	}
}

func (c *nodeCache) removeNode(node *cachedNode) {
	if node.prev != nil {
		node.prev.next = node.next
	} else {
		c.head = node.next
	}
	if node.next != nil {
		node.next.prev = node.prev
	} else {
		c.tail = node.prev
	}
	node.prev = nil
	node.next = nil
}

func (c *nodeCache) moveToHead(node *cachedNode) {
	if c.head == node {
		return
	}
	c.removeNode(node)
	c.addToHead(node)
}

func (c *nodeCache) evictTail() {
	if c.tail == nil {
		return
	}
	node := c.tail
	c.removeNode(node)
	delete(c.items, node.nodeID)
}
