use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// 优先队列最大堆实现
/// 用于KNN搜索等需要维护Top-K元素的场景
pub struct MaxHeap<T, S> 
where 
    T: PartialOrd + Clone,
    S: PartialOrd + Clone
{
    /// 内部数据存储
    heap: BinaryHeap<HeapItem<T, S>>,
    /// 堆的最大容量
    capacity: usize,
}

/// 堆元素结构，包含元素值和排序分数
#[derive(Clone, PartialEq)]
pub struct HeapItem<T, S> 
where 
    T: PartialOrd + Clone,
    S: PartialOrd + Clone
{
    /// 元素值
    pub item: T,
    /// 排序分数
    pub score: S,
}

impl<T, S> Ord for HeapItem<T, S> 
where 
    T: PartialOrd + Clone,
    S: PartialOrd + Clone
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<T, S> PartialOrd for HeapItem<T, S> 
where 
    T: PartialOrd + Clone,
    S: PartialOrd + Clone
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl<T, S> Eq for HeapItem<T, S> 
where 
    T: PartialOrd + Clone,
    S: PartialOrd + Clone
{}

impl<T, S> MaxHeap<T, S> 
where 
    T: PartialOrd + Clone,
    S: PartialOrd + Clone
{
    /// 创建新的最大堆，指定最大容量
    pub fn new(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            capacity,
        }
    }

    /// 添加元素到堆中
    /// 如果堆已满，则与堆顶元素比较，保留较大的
    pub fn push(&mut self, item: T, score: S) {
        let heap_item = HeapItem { item, score };
        
        if self.heap.len() < self.capacity {
            // 堆未满，直接添加
            self.heap.push(heap_item);
        } else if let Some(mut top) = self.heap.peek_mut() {
            // 堆已满，比较并可能替换堆顶
            if heap_item > *top {
                *top = heap_item;
            }
        }
    }

    /// 获取堆顶元素（最大值）
    pub fn peek(&self) -> Option<(&T, &S)> {
        self.heap.peek().map(|item| (&item.item, &item.score))
    }

    /// 弹出堆顶元素
    pub fn pop(&mut self) -> Option<(T, S)> {
        self.heap.pop().map(|item| (item.item, item.score))
    }

    /// 获取堆中元素数量
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// 检查堆是否为空
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// 清空堆
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// 获取堆中所有元素，按优先级排序（从大到小）
    pub fn into_sorted_vec(self) -> Vec<(T, S)> {
        self.heap.into_sorted_vec()
            .into_iter()
            .map(|item| (item.item, item.score))
            .collect()
    }
}

// 其他工具函数可以根据需要在这里添加 