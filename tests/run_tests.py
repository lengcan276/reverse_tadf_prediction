#!/usr/bin/env python
"""运行所有测试"""

import os
import sys
import time

def run_all_tests():
    """运行所有测试模块"""
    
    print("="*60)
    print("RUNNING COMPLETE TEST SUITE FOR TADF/rTADF PREDICTION")
    print("="*60)
    
    test_modules = [
        ('Data Preprocessing', 'tests/test_preprocessing.py'),
        ('Feature Engineering', 'tests/test_features.py'),
        ('Model Training', 'tests/test_models.py'),
    ]
    
    results = []
    
    for name, test_file in test_modules:
        print(f"\n{'='*40}")
        print(f"Testing: {name}")
        print(f"{'='*40}")
        
        start_time = time.time()
        
        try:
            exec(open(test_file).read())
            status = "✓ PASSED"
            results.append((name, status, time.time() - start_time))
        except Exception as e:
            status = f"✗ FAILED: {str(e)}"
            results.append((name, status, time.time() - start_time))
            print(f"\nError in {name}: {e}")
    
    # 打印测试报告
    print("\n" + "="*60)
    print("TEST REPORT")
    print("="*60)
    
    for name, status, duration in results:
        print(f"{name:.<30} {status} ({duration:.2f}s)")
    
    passed = sum(1 for _, s, _ in results if "PASSED" in s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")

if __name__ == "__main__":
    run_all_tests()