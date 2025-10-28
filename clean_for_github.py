"""
GitHub上传前的敏感信息清理脚本
运行此脚本自动清理敏感信息并准备上传

使用方法：
    python clean_for_github.py
"""
import os
import shutil
import re
from pathlib import Path


class GitHubCleaner:
    """GitHub上传准备清理器"""
    
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.sensitive_patterns = [
            r'484807978@qq\.com',
            r'brQZ3p71M68SE',
            r'JZ27229',
        ]
        self.issues_found = []
        
    def check_sensitive_info(self):
        """检查敏感信息"""
        print("=" * 60)
        print("🔍 检查敏感信息...")
        print("=" * 60)
        
        python_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in python_files:
            # 跳过清理脚本自己
            if py_file.name == "clean_for_github.py":
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern in self.sensitive_patterns:
                    if re.search(pattern, content):
                        self.issues_found.append({
                            'file': str(py_file.relative_to(self.root_dir)),
                            'pattern': pattern,
                            'type': 'sensitive_info'
                        })
                        print(f"❌ 发现敏感信息: {py_file.relative_to(self.root_dir)}")
                        print(f"   模式: {pattern}")
            except Exception as e:
                print(f"⚠️ 无法读取文件 {py_file}: {e}")
        
        if not self.issues_found:
            print("✅ 未发现敏感信息")
        
        return len(self.issues_found) == 0
    
    def backup_config(self):
        """备份配置文件"""
        print("\n" + "=" * 60)
        print("💾 备份配置文件...")
        print("=" * 60)
        
        config_file = self.root_dir / "config.py"
        backup_file = self.root_dir / "config_local.py"
        
        if config_file.exists():
            if not backup_file.exists():
                shutil.copy2(config_file, backup_file)
                print(f"✅ 已备份: config.py → config_local.py")
            else:
                print("ℹ️  备份已存在: config_local.py")
        else:
            print("⚠️ config.py 不存在")
    
    def replace_with_template(self):
        """用模板替换配置文件"""
        print("\n" + "=" * 60)
        print("🔄 替换为模板配置...")
        print("=" * 60)
        
        config_file = self.root_dir / "config.py"
        template_file = self.root_dir / "config.py.example"
        
        if template_file.exists():
            shutil.copy2(template_file, config_file)
            print("✅ 已用模板替换 config.py")
        else:
            print("❌ 模板文件 config.py.example 不存在！")
    
    def check_gitignore(self):
        """检查.gitignore配置"""
        print("\n" + "=" * 60)
        print("📋 检查 .gitignore...")
        print("=" * 60)
        
        gitignore_file = self.root_dir / ".gitignore"
        
        required_patterns = [
            "config_local.py",
            ".env",
            "data/raw/*.csv",
            "checkpoints/*.pt",
        ]
        
        if not gitignore_file.exists():
            print("❌ .gitignore 文件不存在！")
            return False
        
        content = gitignore_file.read_text(encoding='utf-8')
        
        all_good = True
        for pattern in required_patterns:
            if pattern in content:
                print(f"✅ {pattern}")
            else:
                print(f"❌ 缺少: {pattern}")
                all_good = False
        
        return all_good
    
    def check_personal_data(self):
        """检查个人数据文件"""
        print("\n" + "=" * 60)
        print("📁 检查个人数据文件...")
        print("=" * 60)
        
        data_dirs = [
            self.root_dir / "data" / "raw",
            self.root_dir / "data" / "preprocessed",
            self.root_dir / "checkpoints",
        ]
        
        large_files = []
        
        for data_dir in data_dirs:
            if data_dir.exists():
                for file in data_dir.rglob("*"):
                    if file.is_file():
                        size_mb = file.stat().st_size / (1024 * 1024)
                        if size_mb > 10:  # 大于10MB的文件
                            large_files.append({
                                'file': str(file.relative_to(self.root_dir)),
                                'size': f"{size_mb:.2f} MB"
                            })
        
        if large_files:
            print("⚠️ 发现大文件（将被.gitignore排除）:")
            for item in large_files:
                print(f"   - {item['file']}: {item['size']}")
        else:
            print("✅ 未发现大文件")
    
    def generate_report(self):
        """生成检查报告"""
        print("\n" + "=" * 60)
        print("📊 GitHub上传准备报告")
        print("=" * 60)
        
        if self.issues_found:
            print("\n❌ 发现以下问题需要处理：\n")
            for issue in self.issues_found:
                print(f"  文件: {issue['file']}")
                print(f"  问题: {issue['type']}")
                print(f"  详情: {issue.get('pattern', 'N/A')}\n")
            return False
        else:
            print("\n✅ 所有检查通过！可以上传到GitHub")
            print("\n📝 下一步:")
            print("  1. 运行: git status")
            print("  2. 运行: git add .")
            print("  3. 运行: git commit -m 'Initial commit'")
            print("  4. 运行: git push")
            return True
    
    def run(self, auto_clean=False):
        """执行完整的检查和清理流程"""
        print("\n" + "🚀 " * 15)
        print("GitHub上传准备工具")
        print("🚀 " * 15 + "\n")
        
        # 1. 检查敏感信息
        safe = self.check_sensitive_info()
        
        # 2. 备份配置
        self.backup_config()
        
        # 3. 如果发现敏感信息且允许自动清理
        if not safe and auto_clean:
            self.replace_with_template()
            print("\n✅ 已自动清理敏感信息")
        elif not safe:
            print("\n⚠️ 发现敏感信息但未自动清理")
            print("   运行 'python clean_for_github.py --auto-clean' 自动清理")
        
        # 4. 检查.gitignore
        self.check_gitignore()
        
        # 5. 检查个人数据
        self.check_personal_data()
        
        # 6. 生成报告
        success = self.generate_report()
        
        return success


def main():
    import sys
    
    auto_clean = '--auto-clean' in sys.argv or '-a' in sys.argv
    
    cleaner = GitHubCleaner()
    success = cleaner.run(auto_clean=auto_clean)
    
    if not success:
        print("\n⚠️ 请先解决上述问题再上传到GitHub！")
        sys.exit(1)
    else:
        print("\n" + "🎉 " * 15)
        print("准备完成！祝上传顺利！")
        print("🎉 " * 15 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
