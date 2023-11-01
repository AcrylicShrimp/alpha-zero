use std::{
    alloc::Layout,
    collections::HashSet,
    mem::{align_of, replace, size_of},
    ptr::NonNull,
};

/// A node allocator that allocates nodes in pages.
pub struct NodeAllocator<T> {
    freed: Vec<NonNull<T>>,
    active_page: Page<T>,
    pages: Vec<Page<T>>,
}

impl<T> NodeAllocator<T> {
    /// Page size; number of nodes per page.
    pub const PAGE_SIZE: usize = 1024;

    /// Create a new node allocator.
    pub fn new() -> Self {
        Self {
            freed: Vec::new(),
            active_page: Page::new(Self::PAGE_SIZE),
            pages: Vec::new(),
        }
    }

    /// Allocate a new node with the given value.
    pub fn allocate(&mut self, value: T) -> NonNull<T> {
        // reuse a freed node if possible
        if let Some(ptr) = self.freed.pop() {
            unsafe { ptr.as_ptr().write(value) };
            return ptr;
        }

        if self.active_page.is_full() {
            // allocate a new page and make it active
            let old_page = replace(&mut self.active_page, Page::new(Self::PAGE_SIZE));
            self.pages.push(old_page);
        }

        let ptr = self.active_page.allocate();
        unsafe { ptr.as_ptr().write(value) };
        ptr
    }

    /// Free the given node. This triggers the destructor of the node.
    pub fn deallocate(&mut self, ptr: NonNull<T>) {
        unsafe { ptr.as_ptr().drop_in_place() };
        self.freed.push(ptr);
    }
}

impl<T> Drop for NodeAllocator<T> {
    fn drop(&mut self) {
        // prevent double frees
        let already_freed = HashSet::from_iter(self.freed.iter().copied());

        self.active_page.drop_all(&already_freed);

        for page in &mut self.pages {
            page.drop_all(&already_freed);
        }
    }
}

/// A single page of memory.
struct Page<T> {
    memory: NonNull<T>,
    capacity: usize,
    offset: usize,
}

impl<T> Page<T> {
    /// Allocate a new page with the given capacity.
    pub fn new(capacity: usize) -> Self {
        debug_assert!(capacity != 0);

        let layout = Layout::from_size_align(capacity * size_of::<T>(), align_of::<T>()).unwrap();
        let memory = unsafe { std::alloc::alloc(layout) };

        if memory.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        Self {
            memory: unsafe { NonNull::new_unchecked(memory as *mut T) },
            capacity,
            offset: 0,
        }
    }

    /// Check if the page is full.
    pub fn is_full(&self) -> bool {
        self.offset == self.capacity
    }

    /// Allocate a new object on the page.
    pub fn allocate(&mut self) -> NonNull<T> {
        debug_assert!(self.offset < self.capacity);

        let ptr = unsafe { self.memory.as_ptr().offset(self.offset as isize) };
        self.offset += 1;
        unsafe { NonNull::new_unchecked(ptr) }
    }

    /// Drop all objects on the page. This triggers the destructor of each object.
    fn drop_all(&mut self, already_freed: &HashSet<NonNull<T>>) {
        for offset in 0..self.offset {
            let ptr =
                unsafe { NonNull::new_unchecked(self.memory.as_ptr().offset(offset as isize)) };

            if !already_freed.contains(&ptr) {
                unsafe { ptr.as_ptr().drop_in_place() };
            }
        }
    }
}

impl<T> Drop for Page<T> {
    fn drop(&mut self) {
        let layout =
            Layout::from_size_align(self.capacity * size_of::<T>(), align_of::<T>()).unwrap();
        unsafe { std::alloc::dealloc(self.memory.as_ptr() as *mut u8, layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate() {
        let mut allocator = NodeAllocator::new();

        let ptr = allocator.allocate(42);
        assert_eq!(unsafe { *ptr.as_ptr() }, 42);
    }
}
