workflow "Unit Test" {
  on = "push"
  resolves = ["action-gtest"]
}

action "action-gtest" {
  uses = "CyberZHG/github-action-gtest@master"
  args = "-d . -e test_basic_primitives"
}
