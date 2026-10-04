# gpt-4o-mini（两阶段 ILR 流水线）首次生成失败题目的阶段归因分析

## 1. 数据来源与范围

- 分析对象：`output/{Algorithm,HumanEval-V,MATH}/gpt-4o-mini-gpt-4o-mini/` 三个文件夹中外层 `samples.jsonl_results.jsonl` 里 `passed == False` 的记录，即**直接生成（第一次生成）没有通过测试**的题目，共 81 题。
- **不包含** `reflection/` 文件夹里反思修复后的结果。
- 每题的完整数据（题面 prompt、ILR、生成代码、失败信息、测试、OCR 文本）和分析结论存放在 `output/stage_analysis_gpt/<数据集>/<题目名>.json` 的 `analysis` 字段。
- 归因口径：`error_source` 取值 **ilr**（ILR 本身逻辑错误，忠实实现也过不了测试）/ **code**（ILR 正确，代码没有忠实或正确实现）/ **both**（ILR 错且代码还额外偏离了它）/ **other**（两者都没错仍失败）。判定基于：忠实转写 ILR 实际运行 + 生成代码实跑测试 + 与 OCR 原始节点/边文本对照。

## 2. 总体统计

| 数据集 | 题目总数 | 首次生成失败 | 首次通过率 | ilr | code | both | other |
|---|---|---|---|---|---|---|---|
| Algorithm | 149 | 37 | 75.2% | 28 | 2 | 7 | 0 |
| HumanEval-V | 164 | 21 | 87.2% | 12 | 4 | 5 | 0 |
| MATH | 125 | 23 | 81.6% | 12 | 1 | 10 | 0 |
| **合计** | **438** | **81** | **81.5%** | **52** | **7** | **22** | **0** |

**结论：81 道失败题中，ILR 阶段错误占主导**——纯 ILR 错误 52 题（64%），ILR+代码双错 22 题（27%），纯代码错误仅 7 题（9%）。即 91% 的失败可追溯到 ILR 生成阶段，只有 9% 是 ILR 正确但代码转写失败。

- ILR 质量分布：incorrect 74 / partially_correct 0 / correct 7；代码相对 ILR：deviates 45 / faithful 36。
- ILR 出错的 74 题中，OCR 贡献判断：OCR 误读/漏致错 2 题（yes）、流程图或 OCR 无法区分 9 题（unclear）、OCR 无责（ILR 环节或原图自身问题）63 题（no）。

## 3. 高频失败模式

### ILR 生成阶段（占绝大多数）

1. **决策节点 Yes/No 分支接反**（最多见）：如 check-if-array-is-good 把 len==1→Return False 接成 True；cell-reachable-at-time、string-transformation、379-minimum-moves、3312-Sorted-GCD 等同型错误。多题可通过 OCR 边证明原图连线正确、错在 ILR 生成环节。
2. **循环回边/出口丢失或错接**：产生死循环、孤儿节点、不可达输出（如 minimum-size-subarray、ball-passing-game、227-Basic-Calculator、2709-GCD-Traversal、k-Mirror-Numbers）。
3. **丢失或改写关键语句/节点**：如 Digit-One 丢掉两句累加（补回即全对）、Abbreviating-Product 丢进位出口、HumanEval-114 丢 min_sum 初始化、HumanEval-103 把二进制转换写成十进制加前缀。
4. **数据结构降级**：SortedList 被写成普通 list、sl.add 误写为 append（continuous-subarrays、divide-array-minimum-cost-ii），Counter 降级为 dict（3312）。
5. **流程图/原图本身错误或残缺被忠实继承**：如 HumanEval-125（replace(',','') 原图即错）、HumanEval-37（循环必然越界）、HumanEval-124（缺格式校验规则）。

### 代码生成阶段（较少）

1. **未按 ILR 提前终止**：HumanEval-12 把"首个最长串即返回"写成循环覆盖取最后一个。
2. **丢弃 ILR 的辅助函数结构**：Pow(x,n) 把快速幂 helper 并进主函数造成无限递归；HumanEval-39 重建算法却把 is_prime 命名错触发 NameError。
3. **细节翻译错误**：正则 r[.?!]\s* 写成双反斜杠（HumanEval-91）、丢 hash_map 下标 -1 偏移导致越界（1399）、rstrip 提前到开头（HumanEval-99）、append 放错分支（HumanEval-123）。

### OCR 的贡献

OCR 直接致错的确定案例很少（如 HumanEval-140 把空格误读为问号、378 的表达式乱码）；更多 ILR 错误发生在 OCR 输入基本正确的情况下，属于 ILR 生成环节自身的接线/逻辑错误。部分题（368、553、780、379 等）OCR 边混乱与原图缺陷无法切分，已标注 unclear。

## 4. 全部 81 题明细

### Algorithm（37 题）

| 题目 | 难度 | error_source | ILR 判定 | 代码 vs ILR | 归因说明 |
|---|---|---|---|---|---|
| biweekly-contest-108-longest-alternating-subarray | Easy | ILR 错误 | incorrect | faithful | ILR 把外层循环边界误写成 i < len(nums)-1（且缺 i 的初始化与递增节点），代码忠实翻译后对末尾才开始的交替子数组漏判，失败源于 ILR 阶段对流程图标签的错译。 |
| biweekly-contest-108-partition-string-into-minimum-beauti... | Medium | ILR 错误 | incorrect | deviates | 流程图本身的递推就写错（OCR 文本清晰显示 dp[j]，应为 dp[i]），ILR 忠实继承该错误且重编号导致连线全面错乱，代码重组控制流后仍带着错误递推，失败根源在流程图/ILR 的算法错误。 |
| biweekly-contest-109-check-if-array-is-good | Easy | ILR 错误 | incorrect | faithful | 流程图连线本身是对的（len==1 → Return False），ILR 生成时把该分支接反成 Return True，代码忠实继承，14 条测试因此失败；失败源于 ILR 连线错误。 |
| biweekly-contest-110-minimum-time-to-make-array-sum-at-mo... | Hard | ILR 错误 | incorrect | faithful | 流程图自身的 DP 转移就是错的（OCR 文本清晰显示 ind[j-1]，物品应取外层第 i 个排序元素而非内层第 j 个），ILR 忠实继承，代码忠实翻译，导致约1/3 用例答案偏大，失败源于 ILR/流程图算法本身。 |
| biweekly-contest-111-number-of-beautiful-integers-in-the-... | Hard | ILR 错误 | incorrect | faithful | 流程图的基例结构本是对的（OCR 边 11→12 证实），ILR 生成时把判断拆散错位（入口处即返回 1、末位无条件返回 1），代码忠实继承后对任何输入都返回 0，失败源于 ILR 连线/逻辑错误。 |
| biweekly-contest-112-count-k-subsequences-of-a-string-wit... | Hard | 两者都有 | incorrect | deviates | 流程图本身就不完整（OCR 节点里没有任何前进到下一频次组的步骤，ILR 忠实继承该缺陷），代码又额外把循环删成单次执行，两层错误叠加：ILR 忠实执行在 'abbcd' 上错（16 vs 2），代码在 'bcca' 等用例上错（2 vs 4）。 |
| biweekly-contest-114-minimum-operations-to-collect-elements | Easy | ILR 错误 | incorrect | faithful | 流程图本身就写错出去重目标（OCR 清晰显示节点7/9 操作的是 nums[i] 而非按值下标 nums[j]），ILR 与代码忠实继承，遇到 needed 值重复时提前计数导致答案偏小，失败源于流程图/ILR 算法本身。 |
| biweekly-contest-114-split-array-into-maximum-number-of-s... | Medium | ILR 错误 | incorrect | deviates | 流程图本身的算法大体正确（OCR 边显示 m==0 判断在循环后、No→Return 1），是 ILR 生成阶段把连线接错（判断进循环、分支接反、Return 1 孤立），代码又按错误结构实现并叠加自己的改动，100 条测试错 91 条；首要错误源为 ILR。 |
| biweekly-contest-115-last-visited-integers | Easy | ILR 错误 | incorrect | faithful | 流程图自身的 'Is ci == -1?' 判据就不充分（ci 减到 -2 以下后仍走 ov[ci] 分支），OCR 文本清晰、ILR 与代码都忠实继承，遇到连续 prev 多于已见整数+1 时崩溃或取错值，失败源于流程图/ILR 算法缺陷。 |
| biweekly-contest-118-maximize-area-of-square-hole-in-grid | Medium | ILR 错误 | incorrect | faithful | 流程图自身的 prev 更新时机就错了（每轮覆盖而非段断裂时更新），ILR 忠实继承、代码忠实翻译，算法退化为恒等于 4，失败源于流程图/ILR 算法本身。 |
| biweekly-contest-118-minimum-number-of-coins-for-fruits | Medium | ILR 错误 | incorrect | faithful | 流程图的控制流本是对的（OCR 边 7-Yes→8→10、7-No→9→10），ILR 生成时把 pop 后的走向错接成回循环头，代码忠实继承该错误，首个测试输入 [3,1,2] 在 i=0 即崩溃，失败源于 ILR 连线错误。 |
| biweekly-contest-120-count-the-number-of-incremovable-sub... | Easy | 两者都有 | incorrect | deviates | ILR 把本已残缺的流程图进一步接乱（死循环、输出挂错分支、next 悬空），代码又抛开 ILR 自行重组出错误的计数语义，两层独立出错：ILR 忠实执行会死循环，代码对示例2 输出 9（期望 7）。 |
| biweekly-contest-122-divide-an-array-into-subarrays-with-... | Hard | ILR 错误 | incorrect | faithful | 流程图/ILR 的算法本身就是错的（cursum 维护逻辑不成立），ILR 又在动作层把 sl.add 误写成 sl.append（与自身标签和 OCR 文本矛盾），代码忠实继承后首个测试即抛 NotImplementedError，失败源于 ILR 阶段。 |
| weekly-contest-352-continuous-subarrays | Medium | ILR 错误 | incorrect | deviates | ILR阶段出错：未把SortedList翻译成可执行语义（用了普通list），且控制流本身残缺（无while重判、悬挂节点引用）；代码虽然把控制流修复成while循环，但继承了普通list判断min/max的致命缺陷，答案错误。 |
| weekly-contest-354-length-of-the-longest-valid-substring | Hard | 两者都有 | incorrect | deviates | ILR阶段把流程图的分支/汇聚边接错（false分支直达res更新并无条件收缩right、循环变量无自减），代码生成阶段又把res更新与right收缩的时机重排，引入'用未收缩right计res'与'right越界增长'两个独立错误，两级都错。 |
| weekly-contest-357-maximum-elegance-of-a-k-length-subsequ... | Hard | ILR 错误 | incorrect | faithful | ILR阶段把流程图菱形8的Yes/No分支接反（OCR标签本身正确），使seenCats/dups语义完全颠倒；代码忠实执行了这套错误ILR，答案必然错误。 |
| weekly-contest-359-determine-the-minimum-sum-of-a-k-avoid... | Medium | ILR 错误 | incorrect | deviates | ILR阶段出错：除自身接线错误（第二段循环不可达、false分支返回i+k）外，还继承了流程图本身不完整的算法（第二段循环从1开始漏加k，仅选n-1个数）；代码虽然修复了ILR的输出与控制流问题，但保留了漏加k的循环定义，答案偏小。 |
| weekly-contest-360-furthest-point-from-origin | Easy | ILR 错误 | incorrect | deviates | ILR/流程图算法本身不完整：下划线处理缺s==t分支且ILR字面控制流中该逻辑不可达（节点11提前return）；代码虽修复了控制流使s!=t情形正确，但保留了s==t时丢弃下划线的缺陷，答案偏小。 |
| weekly-contest-360-maximize-value-of-function-in-a-ball-p... | Hard | ILR 错误 | incorrect | deviates | ILR/流程图算法本身错误：up倍增递推丢字（up[i][j]=up[i][j-1]），导致dp与查询结果错误；ILR还把查询循环出口接错（字面执行死循环）。代码忠实继承递推缺陷并修复控制流后仍因递推错误而答案错误。 |
| weekly-contest-360-minimum-operations-to-form-subsequence... | Hard | ILR 错误 | incorrect | deviates | ILR阶段把流程图明确标注的SortedList/nums.add翻成普通list的sorted()+append，破坏了算法依赖的有序不变量；代码忠实继承该缺陷（仅修复控制流），大量测试操作数偏大。 |
| weekly-contest-361-count-symmetric-integers | Easy | 两者都有 | incorrect | deviates | 两级独立出错：ILR把自增节点改成终点、自增不可达，忠实执行结果恒为0；代码又大幅偏离ILR，给奇数长度数凭空计数，几乎全部测试错误。OCR边本身混乱缺失（6-No->19->4、14-No/17-No缺失），难以判定其误导程度。 |
| weekly-contest-362-determine-if-a-cell-is-reachable-at-a-... | Medium | ILR 错误 | incorrect | faithful | ILR阶段把流程图菱形分支的Yes/No出边接错（OCR边标签本身正确），使d==0与t<ans两个分支的语义互换，代码忠实执行后大量测试真假颠倒。 |
| weekly-contest-362-string-transformation | Hard | ILR 错误 | incorrect | faithful | OCR流程图本身是完全正确的闭式解（8-Yes->10即s==t->f0=1、13-Yes->14即偶数k->left=f0、20-No->21几何公式），ILR生成阶段把各菱形分支接反打乱（f0赋值落到错误路径、left奇偶颠倒、公式节点挂错分支并引用未定义变量）；代码补齐变量后忠实继承了left颠倒的缺陷，绝大多数测试答案错误。 |
| weekly-contest-364-maximum-odd-binary-number | Easy | 两者都有 | incorrect | deviates | ILR逻辑本身错误（'0'追加分支不可达，结果串长度不足），生成的代码又偏离该ILR（自加末位'1'、循环内乱插'0'），两级独立出错。 |
| weekly-contest-365-minimum-size-subarray-in-infinite-array | Medium | ILR 错误 | incorrect | deviates | ILR阶段把循环外的输出选择菱形错误移入循环体当收缩条件、并把输出节点改成循环中途return，两处均为OCR图中不存在的自造接线；代码修复了中途return但忠实继承了永不收缩的门槛，绝大多数测试返回-1。按OCR原图本身也非正确滑动窗口（疑似边误读），但杀死代码的是ILR自造的两处接线。 |
| weekly-contest-366-apply-operations-on-array-to-maximize-... | Hard | ILR 错误 | incorrect | faithful | 流程图/ILR 算法本身设计错误：IDX[i] 无条件自增使 ans 退化为原数组，忠实实现必错；代码只是忠实转换。OCR 节点文本与边（8→No→10）自洽，未见 OCR 误识别误导。 |
| weekly-contest-368-minimum-number-of-groups-to-create-a-v... | Medium | 代码错误 | correct | deviates | ILR 算法正确（贪心枚举组大小 mn 并用 divmod 判可行性），代码生成阶段把共享 ctx 的辅助函数错误地翻译成独立作用域，导致运行时 NameError。 |
| weekly-contest-368-minimum-sum-of-mountain-triplets-ii | Medium | 代码错误 | correct | deviates | ILR 算法正确（前缀最小+翻转后缀最小+枚举峰值），代码生成阶段错误地把循环后的出口判断移入循环体开头，导致所有输入返回 -1。 |
| weekly-contest-370-maximum-balanced-subsequence-sum | Hard | ILR 错误 | incorrect | faithful | ILR 环节自身出错：提前 return 分支+丢弃节点10合并逻辑+条件漏 '-i'+dict/SortedDict 混用，忠实实现必错或崩溃；OCR 节点文本识别良好（含 '- i' 和节点10合并式），ILR 的错误（含分支极性反转）不能归咎于 OCR，边识别虽显凌乱但 ILR 连其标注都未遵循。 |
| weekly-contest-371-maximum-strong-pair-xor-ii | Hard | ILR 错误 | incorrect | faithful | ILR 生成环节自己出错：OCR 文本里外层位循环和第二判定条件都清晰存在，但 ILR 把外层循环整个丢掉、把强对偶判定条件丢掉，且引用未定义变量 i，忠实实现必然输出错误；代码只是顺从地实现了这个残缺 ILR。 |
| weekly-contest-373-count-beautiful-substrings-ii | Hard | 两者都有 | incorrect | deviates | ILR 环节把正确的流程图连线全面接错（初始化孤岛、四分支错位、循环内 return），代码生成环节又偏离 ILR 自行重构并引入首次插入双自增 bug，两级各自独立出错。 |
| weekly-contest-373-make-lexicographically-smallest-array-... | Medium | ILR 错误 | incorrect | deviates | ILR 环节出错：分组阶段在 OCR 边正确的情况下仍把建组逻辑接丢，赋值阶段循环空转且算法描述本身残缺（每组只赋一个元素）；代码修复接线后实现的仍是这个残缺算法。另注：按 OCR 看，流程图赋值阶段本身就不完整（无外层组循环、14-No 回到自身、输出节点悬在赋值之前），但分组阶段的错接与 OCR 无关，属 ILR 自身错误。 |
| weekly-contest-375-count-subarrays-where-max-element-appe... | Medium | 两者都有 | incorrect | deviates | ILR 丢失了 OCR 边中明确存在的 while 回边（收缩循环塌缩为单步 if），代码又自行改为另一套'遇 max 才计数并消耗边界'的错误算法，两级各自独立出错。 |
| weekly-contest-375-count-tested-devices-after-test-operat... | Easy | ILR 错误 | incorrect | faithful | ILR 生成环节凭空给节点8加了数组递减操作，与 '-ans' 判定形成双重扣减，忠实实现必错；OCR 文本与边均完整清晰，未误导 ILR。 |
| weekly-contest-376-apply-operations-to-maximize-frequency... | Hard | ILR 错误 | incorrect | deviates | 流程图循环文本'For right from 1 to len(sum_vals)-1'本应包含right=n的一轮（参考解为range(1,len(sum_vals))），ILR把它错编码为right<len-1的while且漏掉right初始化并让left/right同时自增；代码修复了部分缺陷但保留了关键的off-by-one。 |
| weekly-contest-378-find-longest-special-substring-that-oc... | Medium | ILR 错误 | incorrect | deviates | OCR文本呈明显乱码（'s[count] + 1]'疑为切片表达式s[count:count+…]的误读，且节点15/16无入边、13/14有双重出边），流程图原貌无法还原；ILR按乱码实现出错误逻辑，代码保留该致命逻辑故同样失败。 |
| weekly-contest-379-minimum-moves-to-capture-the-queen | Medium | ILR 错误 | incorrect | faithful | OCR边与ILR互相矛盾（各自恰好接反一对分支：ILR反接节点7、OCR反接节点9），正确连线应为7-Yes→9、9-Yes→10(输出2)、9-No→11(输出1)、7-No→8(输出2)；真实流程图连线无法确定，但ILR的接线肯定错误，代码忠实执行故同样失败。 |

### HumanEval-V（21 题）

| 题目 | 难度 | error_source | ILR 判定 | 代码 vs ILR | 归因说明 |
|---|---|---|---|---|---|
| HumanEval-101 | - | ILR 错误 | incorrect | faithful | 流程图算法本身有缺陷（逗号未替换为空格直接保留），ILR 忠实反映该错误算法，代码也忠实实现，导致按空白 split 后逗号残留。 |
| HumanEval-103 | - | ILR 错误 | incorrect | faithful | ILR 在'Convert average to binary'节点自行采用了错误的转换实现（十进制加前缀），求和与取平均逻辑均正确，代码也忠实实现，导致输出如 '0b3' 而非 '0b11'。 |
| HumanEval-108 | - | 两者都有 | incorrect | deviates | ILR把主流程循环/分支接线错误（判断false分支误接end、决策被移出循环），本身逻辑即错；代码生成又独立犯错——辅助函数误命名为count_nums导致NameError，两边都出了问题。 |
| HumanEval-111 | - | ILR 错误 | incorrect | faithful | ILR 生成环节把'字母的出现次数'错误翻译成 word.count(word)，导致 t 恒为1、输出所有字母计1，代码忠实实现该错误逻辑；OCR 原文只是自然语言'count of letter'，未误导。 |
| HumanEval-114 | - | 两者都有 | incorrect | deviates | ILR生成时漏掉了'min_sum = -max_sum'节点导致输出变量未定义，本身即错；代码生成又独立偏离——将max(-i for i in nums)同时填入两个分支、丢失取负，两边各自出错。 |
| HumanEval-12 | - | 代码错误 | correct | deviates | ILR正确（首个最长字符串即返回），代码生成把end节点的立即返回改写为循环结束后的最后匹配返回，违反题目'并列时返回第一个'的要求。 |
| HumanEval-123 | - | 代码错误 | correct | deviates | ILR 正确（对每个新n判断奇偶后再追加），代码生成把追加语句错误地放进3n+1分支内部，既追加了偶数又漏掉了折半产生的奇数。 |
| HumanEval-124 | - | ILR 错误 | incorrect | faithful | 流程图算法本身就不完整（按OCR文本看只有空串、月份范围、各月天数检查，完全没有格式校验节点），ILR忠实继承该缺陷，代码也忠实实现，遇非法格式直接崩溃。 |
| HumanEval-125 | - | ILR 错误 | incorrect | faithful | 流程图节点本身写的就是 txt.replace(',', '').split()（OCR原文即如此，逗号替换为空串），算法在流程图层面就错了；ILR 忠实照抄，代码也忠实实现。 |
| HumanEval-132 | - | ILR 错误 | incorrect | deviates | ILR生成环节把流程图13号判断的Yes/No边接反且丢失计数节点，按ILR逻辑恒返回False；代码虽修正了计数分支却沿用了ILR错误的'失配即返回False'，仍无法通过要求中途失配后继续统计的用例。OCR边显示流程图本身接线正确（13 Yes→14→12回边，No→15 break→16），OCR无误。 |
| HumanEval-137 | - | ILR 错误 | incorrect | faithful | ILR 在转录输出节点时把'返回原始a/b'误写成'返回替换后的temp_a/temp_b'，代码忠实实现，导致字符串型结果丢失原格式（'2,3'变'2.3'）；OCR原文清晰区分temp替换节点与输出a/b节点，OCR无误。 |
| HumanEval-140 | - | 两者都有 | incorrect | deviates | OCR把流程图中的空格误读为'?'、'-'误读为'!'/'.'且丢失循环回边，ILR在此基础上又把分支接错（空格分支做追加、非空格分支直接终止），忠实执行必然出错；生成代码又偏离了该ILR并自带错误，两边独立造成失败。 |
| HumanEval-155 | - | ILR 错误 | incorrect | faithful | ILR生成阶段把'for each digit in abs(num)'实现为while abs_num>0的数值循环，天然遗漏num=0的情形；代码忠实实现该ILR后在candidate(0)处失败。 |
| HumanEval-156 | - | ILR 错误 | incorrect | faithful | 按OCR文本看流程图本身就把'Decrement div'连回外层'Is number>0'判断、缺少内层重复循环（也可能是OCR误读节点14/15两条箭头的指向，无法确定）；ILR忠实转录该结构，代码又忠实实现ILR，含重复符号的数字全部出错。 |
| HumanEval-20 | - | ILR 错误 | incorrect | faithful | 按OCR文本看流程图本身算法就不完整（缺外层循环回边、节点13提前连到输出、标签closest_pair=sorted([elem,elem2])缺少tuple()转换）；ILR重建后输出节点不可达且输出list，代码忠实实现该逻辑，所有断言因list!=tuple失败。 |
| HumanEval-37 | - | ILR 错误 | incorrect | faithful | OCR边集完全没有循环回边且判断节点位置与分支语义混乱（追加节点7→判断8，Yes反而指向'添加剩余偶元素'），无法确定是流程图本身画错还是OCR漏边；ILR重建时循环条件写错并让节点9不可达，忠实实现即越界崩溃，代码行为与ILR等价。 |
| HumanEval-39 | - | 两者都有 | incorrect | deviates | OCR节点与边完整、按其原编号读图即可得到正确算法，ILR在重编号节点时把边整体接错（ILR生成环节自身错误）；生成代码虽按正确控制流重写（偏离了错误ILR），却把is_prime错命名为prime_fib导致运行即NameError，两处独立造成失败。 |
| HumanEval-62 | - | ILR 错误 | incorrect | faithful | 按OCR文本看流程图本身就没有体现'导数需去掉常数项/从一次项开始'（标签就是enumerate(xs)），ILR生成时又把循环条件合并成logic=True造成死循环；代码忠实实现表意算法后继承i=0缺陷，含非零常数项的断言全部失败。 |
| HumanEval-91 | - | 代码错误 | correct | deviates | ILR正确（re.split(r'[.?!]\s*', S) 配合 startswith('I ') 可通过全部测试），代码生成阶段转写正则时多转义了一层反斜杠导致切分失效，仅Test 5失败。 |
| HumanEval-95 | - | 两者都有 | incorrect | deviates | OCR丢失了#8/#11/#12/#13四个节点的出边（这些节点明显需要出边），ILR据此错误接线成'处理一个key即返回True'的流程；代码虽重写为接近正确的状态机（偏离ILR），却在'非大写非小写即break'后错误地return True，第4个测试失败，两边独立致错。 |
| HumanEval-99 | - | 代码错误 | correct | deviates | ILR正确（仅对含小数点的输入裁剪尾部0，"10"/"0"/"14.5"/"-15.5"/"15.3"均可通过），代码生成阶段错误地调整了语句顺序，把裁剪提前到小数判断之前，破坏了整数字符串的处理。 |

### MATH（23 题）

| 题目 | 难度 | error_source | ILR 判定 | 代码 vs ILR | 归因说明 |
|---|---|---|---|---|---|
| 1330. Reverse Subarray To Maximize Array Value | hard | ILR 错误 | incorrect | faithful | OCR 流程图的两个循环范围本正确（合并才是对的算法），ILR 转写时合并两循环并丢失 i 递增，min2/max2 缺最后一对使内部翻转增益被低估；代码忠实实现该错误结构。 |
| 1399. Count Largest Group | easy | 代码错误 | correct | deviates | ILR 的 -1 下标偏移是正确的，代码生成阶段把它丢了，导致大数 n=9999 时数组越界。 |
| 1716. Calculate Money in Leetcode Bank | easy | ILR 错误 | incorrect | faithful | OCR 文本与边显示流程图真分支本正确（6-Yes→7→9 重置 j=week），ILR 生成时互换节点8/9 语义导致两条分支全错，代码忠实实现故同错。 |
| 1994. The Number of Good Subsets | hard | ILR 错误 | incorrect | faithful | 流程图原本的 DP 嵌套（num 外层）正确，ILR 转写时拆散循环并丢失跳过边造成重复计数；代码忠实实现该错误结构。 |
| 2081. Sum of k-Mirror Numbers | hard | ILR 错误 | incorrect | faithful | ILR 丢掉 OCR 中明确存在的进位出口节点是决定性错误，基-k 回文枚举卡死/跳跃；代码忠实实现同样错误。 |
| 2117. Abbreviating the Product of a Range | hard | 两者都有 | incorrect | deviates | OCR 边缺失剥离循环回边，无法区分是流程图画错还是 OCR 边误检（故 unclear），ILR 按残缺边组装且输出节点不可达必错；代码重写时又引入 top12 双重相乘的独立偏差，两方面各自导致失败。 |
| 227. Basic Calculator II | medium | 两者都有 | incorrect | deviates | ILR 缺循环出口、串尾越界必崩；代码又偏离 ILR 加守卫致最后挂起运算被丢弃，两种独立缺陷各自足以导致失败。 |
| 233. Number of Digit One | hard | ILR 错误 | incorrect | faithful | OCR 输入完好（两个节点的文本与连线俱在），ILR 生成阶段丢句是唯一根因；补回两句后全部测试可通过。 |
| 2376. Count Special Integers | hard | ILR 错误 | incorrect | deviates | ILR 违背清晰的 OCR 边把两处分支极性接反，并继承流程图文本的 9-j 公式缺陷；代码只修复其一，失败根源在 ILR。 |
| 2709. Greatest Common Divisor Traversal | hard | ILR 错误 | incorrect | deviates | ILR 违背清晰的 OCR 边错接出口（其图还会死循环），流程图的降序单趟贪心算法本身也错；代码修复接线后实现的仍是错误算法，把应答 True 的用例判成 False。 |
| 2790. Maximum Number of Groups With Increasing Length | hard | 两者都有 | incorrect | deviates | ILR 的 poss 图死循环且 sub 构造违背 OCR 文本（丢差分数组）必错；代码又把减 1 目标从'最后 groups 个元素'改成'前 groups 个'并沿用错误 sub，独立产生错误的可行性判断，两方面各自导致失败。 |
| 2818. Apply Operations to Maximize Score | hard | ILR 错误 | incorrect | deviates | 流程图/ILR 的栈结构缺失 while 弹栈语义（赋值后不弹出），左右边界数组全错；代码修复了筛法却忠实继承栈缺陷，失败根源在 ILR 阶段产物。 |
| 282. Expression Add Operators | hard | 两者都有 | incorrect | deviates | OCR所见的流程图本身'expression为空'分支的种子逻辑就有缺陷（节点16-No→29 Append str_op→30 dfs前后value不变→30→15→16回环，数字被重复使用），ILR又错接false分支并丢弃节点29/30；生成代码未按ILR实现而是自行发明重复调用，expression恒为空。 |
| 2949. Count Beautiful Substrings II | hard | ILR 错误 | incorrect | deviates | ILR自身致命错误：主循环每轮把cursum重置为0使前缀平衡无法累积（OCR流程图并无此重置），且sumdict更新逻辑错乱；生成代码虽修复了p循环等其他缺陷，但如实保留了该致命bug，统计结果必然错误。 |
| 3102. Minimize Manhattan Distances | hard | 两者都有 | incorrect | deviates | 流程图/ILR算法本身存在'helper内部排序产生的下标 vs 主流程对原数组切片'的错位缺陷，忠实执行也会在部分样例上给出错误答案；生成代码又把辅助函数错误命名导致get_max_i_j未定义而崩溃。 |
| 3116. Kth Smallest Amount With Single Denomination Combin... | hard | 两者都有 | incorrect | deviates | OCR边显示流程图是正确的容斥+二分（13-Yes→14内层循环、19-Yes→20加、21-Yes→22减、4-No→9输出），ILR把循环机制与符号接线全面错编；生成代码未忠实实现ILR而是部分重建，但偶数子集符号错误独立导致答案偏小。 |
| 3307. Find the K-th Character in String Game II | hard | 两者都有 | incorrect | deviates | ILR把循环回边错接到内层判断节点，造成死循环/负移位；生成代码进一步偏离ILR（仅在ops[i]==1时减k、提前return），首个测试即崩溃。 |
| 3312. Sorted GCD Pair Queries | hard | ILR 错误 | incorrect | faithful | OCR边（17-No→18赋值、17-Yes→19跳过）配合Counter语义的流程图本是对的，ILR反转了该分支并把Counter改为dict；生成代码忠实执行了这一错误逻辑，全部gcd计数为0导致下标越界。 |
| 368. Largest Divisible Subset | medium | ILR 错误 | incorrect | faithful | OCR文本#12即为'Update ans[i] = ans[i] + [nums[i]]'，ILR照抄；正确算法应为ans[j]+[nums[i]]，j→i可能是OCR误读也可能是流程图自身缺陷，无法判定，但ILR/代码按此执行必然输出错误子集。 |
| 50. Pow(x, n) | medium | 两者都有 | incorrect | deviates | ILR自身在偶数分支漏赋result，忠实执行也会崩溃；生成代码又整体丢弃辅助函数并把递归调用改成对原函数的无限递归，最终报maximum recursion depth exceeded。 |
| 553. Optimal Division | medium | ILR 错误 | incorrect | faithful | 按OCR文本流程图的输出表达式本就缺括号（也可能是OCR把'/('与')'误读成'/'），ILR与代码忠实照抄，n>=3时全部输出无括号连除式。 |
| 780. Reaching Points | hard | 两者都有 | incorrect | deviates | OCR边本身呈交叉/矛盾连线（3-No→5、4-No→7、5-Yes→6，均非正确连法），真实流程图连线无法确定；ILR按自己的方式错接成无循环直线结构，代码又另行重建但缺特判，两者独立造成失败。 |
| 805. Split Array With Same Average | hard | 两者都有 | incorrect | deviates | OCR流程图算法本身正确（For each a in arr的位掩码子集和+正负分组求交），ILR在转写为索引循环时丢掉了i的初始化；生成代码又把辅助函数错误命名导致possible_sums未定义而崩溃。 |

## 5. 判定口径备注与需人工复核项

- **归因口径偏严**："both" 仅在 ILR 错误之外代码还发生了独立的功能性偏离时使用；若代码只是忠实执行了错误的 ILR，即便有细节改写也归 "ilr"（如 biweekly-114-split-array、HumanEval-132）。若改用"凡偏离即 both"的宽口径，ilr/both 的划分会小幅移动，但 ilr+both 与 code 的总量比不变。
- **ocr_contribution 存疑（unclear）的题目**：biweekly-120-count-incremovable（OCR 缺边无法还原原图）、MATH 2117、368、553、780、379、378、HumanEval-156/37/140、HumanEval-20（若原图含 tuple() 则应改判 OCR 有责）、HumanEval-125（不能排除 OCR 丢空格）。如需定论需回到原始流程图图片核对。
- MATH/2376 的代码在 count=1000 时恰好凑对期望值，属两处偏差抵消的巧合；其测试文件中 9999 的注释值与断言值不一致，但断言本身正确，不影响归因。
