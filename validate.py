#!/usr/bin/env python
"""
Comprehensive validation script for AlgoBrain submission
Tests all components and validates the submission
"""

import sys
import torch
import subprocess
from pathlib import Path

class ValidationReport:
    def __init__(self):
        self.passed = []
        self.failed = []
    
    def add_pass(self, test_name, details=""):
        self.passed.append((test_name, details))
        print(f"✅ {test_name}")
        if details:
            print(f"   {details}")
    
    def add_fail(self, test_name, error):
        self.failed.append((test_name, error))
        print(f"❌ {test_name}")
        print(f"   Error: {error}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        passed = len(self.passed)
        failed = len(self.failed)
        
        print("\n" + "="*60)
        print(f"VALIDATION SUMMARY: {passed}/{total} tests passed")
        print("="*60)
        
        if failed > 0:
            print(f"\n⚠️  {failed} test(s) failed:")
            for name, error in self.failed:
                print(f"  • {name}: {error}")
        else:
            print("\n🎉 All validation tests passed!")
        
        return failed == 0

def validate_openenv_yaml():
    """Validate openenv.yaml exists and structure"""
    report = ValidationReport()
    
    try:
        with open("openenv.yaml", "r") as f:
            content = f.read()
        report.add_pass("openenv.yaml - File readable", f"Size: {len(content)} bytes")
    except Exception as e:
        report.add_fail("openenv.yaml - File read", str(e))
        return report
    
    # Check required fields as strings
    checks = {
        "name:": "name field",
        "version:": "version field",
        "environment:": "environment section",
        "entry_point:": "entry_point",
        "tasks:": "tasks section",
        "easy": "easy task",
        "medium": "medium task",
        "hard": "hard task"
    }
    
    for check, desc in checks.items():
        if check in content:
            report.add_pass(f"openenv.yaml - Contains {desc}")
        else:
            report.add_fail(f"openenv.yaml - Missing {desc}", f"'{check}' not found")
    
    return report

def validate_environment_py():
    """Validate environment.py"""
    report = ValidationReport()
    
    try:
        from environment import AlgoBrainEnv
        report.add_pass("environment.py - Import successful")
    except Exception as e:
        report.add_fail("environment.py - Import", str(e))
        return report
    
    try:
        env = AlgoBrainEnv()
        report.add_pass("environment.py - Instantiation successful")
    except Exception as e:
        report.add_fail("environment.py - Instantiation", str(e))
        return report
    
    try:
        obs, info = env.reset()
        required_keys = {"cpu_temp", "battery_maH", "accuracy_score"}
        if set(obs.keys()) == required_keys:
            report.add_pass("environment.py - reset() returns correct state", f"Keys: {list(obs.keys())}")
        else:
            report.add_fail("environment.py - reset() state keys", f"Expected {required_keys}, got {set(obs.keys())}")
    except Exception as e:
        report.add_fail("environment.py - reset()", str(e))
        return report
    
    try:
        obs, reward, done, truncated, info = env.step(0)
        report.add_pass("environment.py - step() works", f"Reward: {reward:.2f}, Done: {done}")
    except Exception as e:
        report.add_fail("environment.py - step()", str(e))
        return report
    
    return report

def validate_inference_py():
    """Validate inference.py"""
    report = ValidationReport()
    
    try:
        from inference import PolicyNet, get_obs_tensor, train
        report.add_pass("inference.py - Imports successful")
    except Exception as e:
        report.add_fail("inference.py - Imports", str(e))
        return report
    
    try:
        policy = PolicyNet()
        report.add_pass("inference.py - PolicyNet instantiation successful")
    except Exception as e:
        report.add_fail("inference.py - PolicyNet creation", str(e))
        return report
    
    try:
        from environment import AlgoBrainEnv
        env = AlgoBrainEnv()
        obs, _ = env.reset()
        obs_tensor = get_obs_tensor(obs)
        report.add_pass("inference.py - get_obs_tensor() works", f"Tensor shape: {obs_tensor.shape}")
    except Exception as e:
        report.add_fail("inference.py - get_obs_tensor()", str(e))
        return report
    
    try:
        output = policy(obs_tensor)
        report.add_pass("inference.py - Policy forward pass works", f"Output shape: {output.shape}")
    except Exception as e:
        report.add_fail("inference.py - Policy forward pass", str(e))
        return report
    
    return report

def validate_dockerfile():
    """Validate Dockerfile"""
    report = ValidationReport()
    
    try:
        with open("Dockerfile", "r") as f:
            content = f.read()
        report.add_pass("Dockerfile - File exists and readable")
    except Exception as e:
        report.add_fail("Dockerfile - Read", str(e))
        return report
    
    # Check required directives
    checks = {
        "FROM python": "Base image specified",
        "WORKDIR": "Working directory set",
        "COPY": "Files copied to container",
        "RUN pip install": "Dependencies installed",
        "CMD": "Command specified"
    }
    
    for check, desc in checks.items():
        if check in content:
            report.add_pass(f"Dockerfile - Contains '{check}'", desc)
        else:
            report.add_fail(f"Dockerfile - Missing '{check}'", desc)
    
    return report

def validate_requirements_txt():
    """Validate requirements.txt"""
    report = ValidationReport()
    
    try:
        with open("requirements.txt", "r") as f:
            reqs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        report.add_pass("requirements.txt - File readable", f"Contains {len(reqs)} packages")
    except Exception as e:
        report.add_fail("requirements.txt - Read", str(e))
        return report
    
    required_packages = ["torch", "numpy", "requests"]
    for pkg in required_packages:
        if any(pkg in req for req in reqs):
            report.add_pass(f"requirements.txt - Contains '{pkg}'")
        else:
            report.add_fail(f"requirements.txt - Missing '{pkg}'", "Required dependency")
    
    return report

def validate_git_status():
    """Validate git repository status"""
    report = ValidationReport()
    
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            report.add_pass("git - Repository initialized")
        else:
            report.add_fail("git - Not a git repository", result.stderr)
    except Exception as e:
        report.add_fail("git - Check status", str(e))
        return report
    
    return report

def main():
    print("="*60)
    print("🔍 AlgoBrain Submission Validation")
    print("="*60)
    print()
    
    all_reports = []
    
    print("📋 Testing openenv.yaml...")
    print("-" * 40)
    all_reports.append(validate_openenv_yaml())
    print()
    
    print("📋 Testing environment.py...")
    print("-" * 40)
    all_reports.append(validate_environment_py())
    print()
    
    print("📋 Testing inference.py...")
    print("-" * 40)
    all_reports.append(validate_inference_py())
    print()
    
    print("📋 Testing Dockerfile...")
    print("-" * 40)
    all_reports.append(validate_dockerfile())
    print()
    
    print("📋 Testing requirements.txt...")
    print("-" * 40)
    all_reports.append(validate_requirements_txt())
    print()
    
    print("📋 Testing git status...")
    print("-" * 40)
    all_reports.append(validate_git_status())
    print()
    
    # Final summary
    total_passed = sum(len(r.passed) for r in all_reports)
    total_failed = sum(len(r.failed) for r in all_reports)
    
    print("\n" + "="*60)
    print("📊 OVERALL VALIDATION RESULT")
    print("="*60)
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed} ✅")
    print(f"Failed: {total_failed} ❌")
    print("="*60)
    
    if total_failed == 0:
        print("\n🎉 All validation tests passed!")
        print("Your submission is ready for deployment!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} test(s) failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

class ValidationReport:
    def __init__(self):
        self.passed = []
        self.failed = []
    
    def add_pass(self, test_name, details=""):
        self.passed.append((test_name, details))
        print(f"✅ {test_name}")
        if details:
            print(f"   {details}")
    
    def add_fail(self, test_name, error):
        self.failed.append((test_name, error))
        print(f"❌ {test_name}")
        print(f"   Error: {error}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        passed = len(self.passed)
        failed = len(self.failed)
        
        print("\n" + "="*60)
        print(f"VALIDATION SUMMARY: {passed}/{total} tests passed")
        print("="*60)
        
        if failed > 0:
            print(f"\n⚠️  {failed} test(s) failed:")
            for name, error in self.failed:
                print(f"  • {name}: {error}")
        else:
            print("\n🎉 All validation tests passed!")
        
        return failed == 0

def validate_openenv_yaml():
    """Validate openenv.yaml syntax and structure"""
    report = ValidationReport()
    
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
        report.add_pass("openenv.yaml - Valid YAML syntax", str(config.get("name")))
    except Exception as e:
        report.add_fail("openenv.yaml - YAML parsing", str(e))
        return report
    
    # Check required fields
    required_fields = ["name", "version", "environment", "tasks"]
    for field in required_fields:
        if field in config:
            report.add_pass(f"openenv.yaml - Contains '{field}'")
        else:
            report.add_fail(f"openenv.yaml - Missing '{field}'", "Required field not found")
    
    # Validate environment config
    try:
        env = config.get("environment", {})
        if "entry_point" in env:
            report.add_pass("openenv.yaml - Environment entry_point defined")
        if "config" in env or "env_config" in env:
            report.add_pass("openenv.yaml - Environment configuration present")
    except Exception as e:
        report.add_fail("openenv.yaml - Environment config", str(e))
    
    # Validate tasks
    try:
        tasks = config.get("tasks", [])
        if len(tasks) == 3:
            report.add_pass("openenv.yaml - All 3 tasks defined (easy, medium, hard)")
        task_names = [t.get("name") for t in tasks]
        for name in ["easy", "medium", "hard"]:
            if name in task_names:
                report.add_pass(f"openenv.yaml - Task '{name}' present")
    except Exception as e:
        report.add_fail("openenv.yaml - Tasks validation", str(e))
    
    return report

def validate_environment_py():
    """Validate environment.py"""
    report = ValidationReport()
    
    try:
        from environment import AlgoBrainEnv
        report.add_pass("environment.py - Import successful")
    except Exception as e:
        report.add_fail("environment.py - Import", str(e))
        return report
    
    try:
        env = AlgoBrainEnv()
        report.add_pass("environment.py - Instantiation successful")
    except Exception as e:
        report.add_fail("environment.py - Instantiation", str(e))
        return report
    
    try:
        obs, info = env.reset()
        report.add_pass("environment.py - reset() works", f"State keys: {list(obs.keys())}")
    except Exception as e:
        report.add_fail("environment.py - reset()", str(e))
        return report
    
    try:
        obs, reward, done, truncated, info = env.step(0)
        report.add_pass("environment.py - step() works", f"Reward: {reward:.2f}, Done: {done}")
    except Exception as e:
        report.add_fail("environment.py - step()", str(e))
        return report
    
    return report

def validate_inference_py():
    """Validate inference.py"""
    report = ValidationReport()
    
    try:
        from inference import PolicyNet, get_obs_tensor, train
        report.add_pass("inference.py - Imports successful")
    except Exception as e:
        report.add_fail("inference.py - Imports", str(e))
        return report
    
    try:
        policy = PolicyNet()
        report.add_pass("inference.py - PolicyNet instantiation successful")
    except Exception as e:
        report.add_fail("inference.py - PolicyNet creation", str(e))
        return report
    
    try:
        from environment import AlgoBrainEnv
        env = AlgoBrainEnv()
        obs, _ = env.reset()
        obs_tensor = get_obs_tensor(obs)
        report.add_pass("inference.py - get_obs_tensor() works", f"Tensor shape: {obs_tensor.shape}")
    except Exception as e:
        report.add_fail("inference.py - get_obs_tensor()", str(e))
        return report
    
    try:
        output = policy(obs_tensor)
        report.add_pass("inference.py - Policy forward pass works", f"Output shape: {output.shape}")
    except Exception as e:
        report.add_fail("inference.py - Policy forward pass", str(e))
        return report
    
    return report

def validate_dockerfile():
    """Validate Dockerfile"""
    report = ValidationReport()
    
    try:
        with open("Dockerfile", "r") as f:
            content = f.read()
        report.add_pass("Dockerfile - File exists and readable")
    except Exception as e:
        report.add_fail("Dockerfile - Read", str(e))
        return report
    
    # Check required directives
    checks = {
        "FROM python": "Base image specified",
        "WORKDIR": "Working directory set",
        "COPY": "Files copied to container",
        "RUN pip install": "Dependencies installed",
        "CMD": "Command specified"
    }
    
    for check, desc in checks.items():
        if check in content:
            report.add_pass(f"Dockerfile - Contains '{check}'", desc)
        else:
            report.add_fail(f"Dockerfile - Missing '{check}'", desc)
    
    return report

def validate_requirements_txt():
    """Validate requirements.txt"""
    report = ValidationReport()
    
    try:
        with open("requirements.txt", "r") as f:
            reqs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        report.add_pass("requirements.txt - File readable", f"Contains {len(reqs)} packages")
    except Exception as e:
        report.add_fail("requirements.txt - Read", str(e))
        return report
    
    required_packages = ["torch", "numpy", "requests"]
    for pkg in required_packages:
        if any(pkg in req for req in reqs):
            report.add_pass(f"requirements.txt - Contains '{pkg}'")
        else:
            report.add_fail(f"requirements.txt - Missing '{pkg}'", "Required dependency")
    
    return report

def validate_git_status():
    """Validate git repository status"""
    report = ValidationReport()
    
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        if result.returncode == 0:
            report.add_pass("git - Repository initialized")
        else:
            report.add_fail("git - Not a git repository", result.stderr)
    except Exception as e:
        report.add_fail("git - Check status", str(e))
        return report
    
    return report

def main():
    print("="*60)
    print("🔍 AlgoBrain Submission Validation")
    print("="*60)
    print()
    
    all_reports = []
    
    print("📋 Testing openenv.yaml...")
    print("-" * 40)
    all_reports.append(validate_openenv_yaml())
    print()
    
    print("📋 Testing environment.py...")
    print("-" * 40)
    all_reports.append(validate_environment_py())
    print()
    
    print("📋 Testing inference.py...")
    print("-" * 40)
    all_reports.append(validate_inference_py())
    print()
    
    print("📋 Testing Dockerfile...")
    print("-" * 40)
    all_reports.append(validate_dockerfile())
    print()
    
    print("📋 Testing requirements.txt...")
    print("-" * 40)
    all_reports.append(validate_requirements_txt())
    print()
    
    print("📋 Testing git status...")
    print("-" * 40)
    all_reports.append(validate_git_status())
    print()
    
    # Final summary
    total_passed = sum(len(r.passed) for r in all_reports)
    total_failed = sum(len(r.failed) for r in all_reports)
    
    print("\n" + "="*60)
    print("📊 OVERALL VALIDATION RESULT")
    print("="*60)
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed} ✅")
    print(f"Failed: {total_failed} ❌")
    print("="*60)
    
    if total_failed == 0:
        print("\n🎉 All validation tests passed!")
        print("Your submission is ready for deployment!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} test(s) failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
