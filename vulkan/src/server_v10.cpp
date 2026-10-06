#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>
int main(int argc, char **argv) {
  const char *python = std::getenv("JR_VK_PYTHON");
  if (!python)
    python = "python3";
  auto runtime =
      std::filesystem::absolute(argv[0]).parent_path() / "jr-vk-runtime-v10";
  std::vector<std::string> args{python, JR_V10_FRONTEND, "--runtime",
                                runtime.string(), "--server"};
  for (int i = 1; i < argc; i++)
    args.emplace_back(argv[i]);
  std::vector<char *> ptr;
  for (auto &x : args)
    ptr.push_back(x.data());
  ptr.push_back(nullptr);
  execvp(python, ptr.data());
  std::perror("V10 server Python");
  return 1;
}
