class Solution:
  def __init__(self):
    pass
    
  def binary_search(self, nums, target):
    """二分查找"""
    left, right = 0, len(nums)-1  # 注意right， 左右闭区间
    while left <= right:  # 因为是闭区间，所以<=
        mid = left + (right - left) / 2  # prevent out of bounds
        if nums[mid] == target:
            return mid
        if nums[mid] > target:
            right = mid - 1
        if nums[mid] < target:
            left = mid + 1
    return -1
