// CLAUDE:SUMMARY Thread-safe LRU cache for graph nodes with doubly-linked list eviction.
package horosvec

import "sync"

// cachedNode holds graph node data in the LRU cache.
type cachedNode struct {
	nodeID    int64
	extID     []byte    // external ID for result mapping
	neighbors []int64
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
	items    map[int64]*cachedNode

	// doubly-linked list: head = most recently used, tail = least recently used
	head, tail *cachedNode
}

// newNodeCache creates an LRU cache with the given capacity.
func newNodeCache(capacity int) *nodeCache {
	return &nodeCache{
		capacity: capacity,
		items:    make(map[int64]*cachedNode, capacity),
	}
}

// get retrieves a node from the cache and promotes it in the LRU.
// Returns nil if not found.
func (c *nodeCache) get(nodeID int64) *cachedNode {
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

// getReadOnly retrieves a node without updating LRU order.
// Safe for concurrent readers — only takes a read lock.
// Use on hot search paths where LRU promotion is not needed
// (cache is typically warm and no eviction occurs during search).
func (c *nodeCache) getReadOnly(nodeID int64) *cachedNode {
	c.mu.RLock()
	node := c.items[nodeID]
	c.mu.RUnlock()
	return node
}

// put adds or updates a node in the cache.
func (c *nodeCache) put(node *cachedNode) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if existing, ok := c.items[node.nodeID]; ok {
		// Update existing
		existing.extID = node.extID
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
	c.items = make(map[int64]*cachedNode, c.capacity)
	c.head = nil
	c.tail = nil
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
