#pragma once
#include "crt_utils.hpp"

struct CRTProductTree {
  std::vector<std::vector<cpp_int>> levels;
};

inline CRTProductTree build_product_tree(const std::vector<u32>& m) {
  CRTProductTree T;
  T.levels.emplace_back();
  T.levels.back().reserve(m.size());
  for (u32 mi : m) T.levels.back().emplace_back(cpp_int(mi));
  while (T.levels.back().size() > 1) {
    const auto& cur = T.levels.back();
    std::vector<cpp_int> nxt; nxt.reserve((cur.size()+1)/2);
    for (size_t i=0;i<cur.size(); i+=2) {
      nxt.push_back(i+1<cur.size() ? cur[i]*cur[i+1] : cur[i]);
    }
    T.levels.push_back(std::move(nxt));
  }
  return T;
}

inline void remainder_tree_down(const CRTProductTree& T, const cpp_int& N,
                                std::vector<cpp_int>& leaf_rems) {
  if (T.levels.empty()) { leaf_rems.clear(); return; }
  const size_t L = T.levels.size();
  std::vector<std::vector<cpp_int>> rems(L);
  rems[L-1].resize(1);
  rems[L-1][0] = N % T.levels[L-1][0];

  for (size_t level=L-1; level>0; --level) {
    const auto& cur_products = T.levels[level-1];
    const auto& parent_rems = rems[level];
    auto& cur_rems = rems[level-1];
    cur_rems.resize(cur_products.size());
    size_t idx_child = 0;
    for (size_t i=0;i<parent_rems.size(); ++i) {
      const cpp_int& R = parent_rems[i];
      cur_rems[idx_child] = R % cur_products[idx_child];
      ++idx_child;
      if (idx_child < cur_products.size()) {
        cur_rems[idx_child] = R % cur_products[idx_child];
        ++idx_child;
      }
    }
  }
  leaf_rems = std::move(rems[0]);
}