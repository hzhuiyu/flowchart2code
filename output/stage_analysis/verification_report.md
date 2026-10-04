# qwen3-vl-8b-instruct（两阶段 ILR 流水线）首次生成失败题目的 MD 分析报告核验

## 1. 核验对象与方法

- 核验对象：`output/stage_analysis/` 下三份报告（`algorithm_ilr_analysis.md`、`humaneval_ilr_analysis.md`、`math_ilr_analysis.md`）对 111 道 `qwen3-vl-8b-instruct-qwen3-vl-8b-instruct` 首次生成失败题的逐题结论（"ILR Error" / "Code Generation Error"）。
- 核验方法（与原报告相互独立）：对每题从题面独立理解题意 → 把 ILR 节点语义忠实转写成 Python 实际运行（对照全部测试）→ 把生成代码逐断言实跑 → 与 OCR 原始节点/边文本对照，再判定真实归因并与 MD 结论比对。
- 判定口径：actual_error_source = **ilr**（ILR 本身逻辑错误，忠实实现也过不了测试）/ **code**（ILR 正确但代码没忠实或正确实现）/ **both**（ILR 错且代码还额外偏离了它）/ **other**。MD 结论评为：**yes**（准确）/ **partial**（根因方向对但不完整或误导，如实际为 both）/ **no**（归因错误）。
- 每题证据存于对应 JSON 的 `verification` 字段。

## 2. 总体结论

| 数据集 | 题数 | 实际 ilr | 实际 both | 实际 code | MD 准确(yes) | 不完整(partial) | 归因错误(no) |
|---|---|---|---|---|---|---|---|
| Algorithm | 51 | 29 | 20 | 2 | 29 | 20 | 2 |
| HumanEval-V | 28 | 16 | 10 | 2 | 17 | 9 | 2 |
| MATH | 32 | 15 | 15 | 2 | 15 | 15 | 2 |
| **合计** | **111** | **60** | **45** | **6** | **61** | **44** | **6** |

**核心发现：**

1. MD 的结论**过半完全准确（61/111，55%）**，但**没有一例被判错方向后仍算准确**——需要修正或补充的共 50 题（45%）。
2. **6 题归因错误（no）**：其中 5 题 MD 标 "ILR Error" 实际是纯代码错误（ILR 忠实转写可通过全部测试），1 题 MD 标 "Code Generation Error" 实际是 both。
3. **44 题不完整（partial）**：MD 根因方向对（ILR 确实有错），但生成代码还额外偏离了 ILR、引入了 ILR 之外的新错误（实际为 both），单一 "ILR Error" 标签掩盖了代码层问题。
4. 真实分布 vs MD 分布：实际 ilr 60 / both 45 / code 6；MD 声称 ILR Error 109 / Code Generation Error 2。**MD 系统性地把代码错误计入了 ILR 错误**：实际纯代码错误 6 题，MD 只识别出 1 题（HumanEval-12）。

## 3. 归因错误的 6 题（需改判）

| 题目 | MD 结论 | 实际归因 | 真实情况 |
|---|---|---|---|
| 2514._Count_Anagrams.json | ILR Error | **code** | 真实失败原因是代码超时而非 ILR：生成代码数学上完全正确（len! / 各字符count! 再取模），实跑全部断言通过，但 'a'*100000 一项就要约 8.3 秒（用 Python 循环两次构造 10 万的阶乘再做bigint除法），在原评测时限内 timed out。ILR 图其实也接错了线（节点7 false 边指向节点10 'Return fact'，使除以字符阶乘的 9/11/12 循环环不可达），忠实执行会得 'too hot'→36（期望18），但生成代码没有沿用这个错误而是按流程图意图补全了除法循环，因此 MD 归因为 ILR Error 不成立。 |
| 780._Reaching_Points.json | ILR Error | **code** | MD 错。ILR 忠实转写通过全部 9 条测试（对样本口径而言 ILR 正确），失败在代码：生成代码把 ILR 的模检查 (ty-sy)%tx==0/(tx-sx)%ty==0 擅自改成 return ty==sy / return tx==sx，偏离 ILR，导致 (1,1,1,1e9)、(1,1,1e9,1) 误判 False。另注：ILR 有未暴露的潜在缺陷——ty==sy 时节点6恒 0%tx==0→True（如 (2,2,3,2) 真值 False 而 ILR 给 True），但测试全部 sx=sy=1，整套测试通过。 |
| HumanEval-108.json | Code Generation Error | **both** | 归因应为 both：ILR 的 digits_sum 在 n>=0 分支从不初始化 neg，node5 直接 n[0]*=ctx['neg']，忠实转写对正数抛 KeyError('neg')，除空数组外所有测试都会崩；同时生成代码确实另有自身致命 bug——把返回值赋给与函数同名的局部变量 digits_sum 造成 UnboundLocalError（与报错吻合），且自行补了 else: neg=1（偏离 ILR 但属修复）。MD 只说代码错，漏掉了 ILR 本身 broken。 |
| HumanEval-39.json | ILR Error | **code** | MD 判错：ILR 算法本身正确（斐波那契递推、遇质数减 n、n 归零输出，补上标准 is_prime 后 10 条断言全过），真实失败原因是代码生成阶段原样保留未定义的 is_prime 调用而未实现它，NameError 崩溃——这是代码生成错误而非 ILR 错误。 |
| biweekly-contest-108-number-of-black-blocks.json | ILR Error | **code** | ILR 描述的算法正确（x/y 从 i-1..i、j-1..j 即 x_end 为开区间端点），忠实转写可通过全部示例；生成代码却写成 range(x_start, x_end+1)，多遍历了 x=i+1、y=j+1，把不含黑格的块也计数，属代码未忠实实现 ILR。 |
| biweekly-contest-109-check-if-array-is-good.json | ILR Error | **code** | ILR 的判定逻辑（硬编码特例+len==1 时 true_next 指向 Return False+一般检查）忠实转写后 100 组测试全部通过；生成代码却把 len(nums)==1 分支写成 return True（ILR 中该分支指向 Return False），导致 [1]、[2] 等全部单元素数组误判，属代码偏离 ILR。 |

## 4. 不完整的 44 题（根因对，但漏了代码侧问题，实际为 both）

| 题目 | 代码额外偏离（ILR 之外的新错误） |
|---|---|
| 2081._Sum_of_k-Mirror_Numbers.json | ILR 确实根本性错误：fn 节点13先把 x[i] 加1，节点14/15 又把 x[n..i]（含 i 本身）全部清零，增量被撤销，fn 永远停在 ['0']，主循环 n 次全加 val=0，任何输入都得 0；且主循环仅迭代 n 次、并非枚举前 n 小的 k-mirror 数。但代码还额外偏离 ILR：ILR 的 fn 逻辑用共享上下文的 ctx['k']（作用域内可解析），生成代码把 fn 写成独立函数却引用裸名 k，直接 NameError 崩溃（即记录的失败原因），故为 both。 |
| 2117._Abbreviating_the_Product_of_a_Range.json | ILR 确实错误且是根因：算法把 c2/c5 当'是否被2/5整除'计数（8 只算 1 个 2），tail5 连 2/5 因子一起乘，末尾还有 range(c2-c5) 额外乘 2/5 的'补偿'循环，完全不是去除尾零的正确方法，忠实实现也无法通过。但生成代码还额外偏离 ILR：ILR 中奇数路径不更新 tail5、c5 只在偶数时检查，偶数 i 的 tail5 要乘两次，代码把这些全部规整成每个 i 各乘一次、c5 对所有 i 检查，两者输出不同，故 both。 |
| 2183._Count_Array_Pairs_Divisible_by_K.json | ILR 确实不可执行：因子循环 6→7→8→6 中 x 在节点6被重置为 1 且无自增，字面执行永远死循环（factors 无限追加 1）；节点14 用到的 f 从未定义；主循环节点 9-12 无任何入边不可达。但生成代码也额外偏离 ILR：没有实现 ILR 的 freq 除数配对逻辑，而是 freq[num]+=1 且精确查找 freq[k//x] 的另一套（错误）算法——coutPairs([1,2,3,4,5],2)=4（期望7）、([2,4,6,8],2)=0（期望6）；且缺 List 导入，def 处就 NameError 崩溃，故 both。 |
| 227._Basic_Calculator_II.json | ILR 确实不可执行：节点3 每次循环都重置 i=s[0]、idx=0，全程无 idx 自增，也没有任何边能跳出循环到达求和节点17/18，字面执行死循环。但生成代码还额外偏离 ILR：ILR 中数字分支节点5 的 next 是节点7（末字符数字仍能触发入栈），代码却让数字分支 continue 跳过运算符/末尾判断，导致表达式最后一个数永远不入栈，故 both。 |
| 264._Ugly_Number_II.json | ILR 确实错误：OCR 中 'ugly = Ist.pop(0)' 这一步在 ILR 里被丢掉，ctx['ugly'] 从未赋值，字面执行立即 KeyError；节点5/6 指向不存在的节点12，节点8/9（排序、返回）不可达，i 也无自增。但生成代码还额外偏离：流程图要求每次取最小元并重新排序（pop+sort 保证有序），代码改成直接取未排序的 Ist[i]，而 Ist 是 BFS 生成顺序并非升序，n≥5 即出错，故 both。 |
| 2709._Greatest_Common_Divisor_Traversal.json | ILR 的贪心算法本身不成立：按降序对每个 i 找第一个 gcd>1 的 j 合并、内层耗尽即返回 False，无法通过合并过的大数回连后续元素，忠实执行在 [2,3,6]→False（期望True）、[4,3,12,8]→False（期望True）就失败（该题正确解法需要质因子并查集）。生成代码又额外偏离 ILR：删掉了节点15 false→16 的'内层耗尽返回 False'，一律落到结尾 return True，导致 [3,9,5]、[100000,50001,25000]、[99991,99989,99971] 都错判为 True，故 both。 |
| 2790._Maximum_Number_of_Groups_With_Increasing_Length.json | ILR 的 poss(groups) 可行性检查是残缺的：两个 'For i in ...' 循环（节点20/23）循环体为空，真正的可行性判断逻辑丢失；按节点20字面语义 i=-groups+1，节点21 对 groups>=2 直接返回 False，poss 恒为 False，二分结果恒为 0，忠实实现全部测试失败。生成代码又额外偏离：把 i=-groups 的空循环补成循环到 0，使 poss 恒为 True，二分结果恒为 n（[2,1,2]→3 期望2、[1,1,1,1,1]→5 期望2）；且缺 List 导入 def 处即 NameError，故 both。 |
| 2818._Apply_Operations_to_Maximize_Score.json | ILR确有错：sieve外层回边指回node8会重置i=2，且node11在内层循环结束后执行prime[j]=False，j>=upper必越界(IndexError)。但代码也没忠实实现ILR：改成prime[i]=False且不标记合数，导致primeScore对合数重复计数（如nums[8]的primeScore算成3而非1），test1输出64≠81。 |
| 29._Divide_Two_Integers.json | ILR确有错：node18回边到node12后，abs_divisor最终会降到0，node12条件dividend>=0恒真导致\|divisor\|=1时死循环（如divide(2**30,1)、divide(-2**31,1)），其余输入结果正确。但代码还额外偏离ILR：完全丢掉node14的'multiple>0'守卫，写成while abs_dividend>=abs_divisor，divisor降为0后0>=0恒真，连ILR能算对的divide(10,3)都死循环，这正是首次失败的直接原因。 |
| 2949._Count_Beautiful_Substrings_II.json | ILR确实错得离谱：node4循环条件用ctx['i']但从未初始化i，node5的j也从未赋值，忠实执行立即KeyError；即便把j修成i，p的翻倍(node7)挂在不可达分支上，p未乘2会导致重复计数（k=2时'baeyh'会算出4而非2）。但代码也完全背离ILR：无视流程图算法，自己写了带大量注释的O(n^2)暴力枚举，小样例能过但大输入必然超时，这正是首次失败的直接原因(timed out)。 |
| 3102._Minimize_Manhattan_Distances.json | ILR确有错：node6把辅助函数返回的(缩减后数组的)下标覆盖写回ctx['i']，node7又据此删点，等于删了两个点求极差，系统性低估答案（300组随机测试237组与暴力解不符；[[1,1],[2,2],[3,3]]算出0期望2）。但代码也偏离ILR：把辅助函数体错误地定义成第二个minimumDistance，真正调用的get_max_i_j从未定义，运行即NameError，这是首次失败的直接报错。 |
| 3154._Find_Number_of_Ways_to_Reach_the_K-th_Stair.json | ILR确实错：外层循环变量row从未初始化（node8求ctx['row']即KeyError），col/i同样未初始化，node12(next=10)在col>row后死循环，且node19直接到node20只累加一项二项式系数（k=1只能得3，期望4）。但代码也大幅偏离ILR：自行补齐循环、target为0/1时提前return 1、只在循环结束后判定一次，结果(0)->1(期望2)、(1)->1(期望4)、(2)->0(期望4)全错。 |
| 3251._Find_the_Count_of_Monotonic_Pairs_II.json | ILR确实错且不可执行：节点id 8重复（process与end同名冲突），node21(i-=1)无任何入边不可达导致内层j循环永远退不出去（忠实执行死循环），j=1000那轮更新被node11整体跳过，diff还用错方向(max(0,nums[i]-nums[i-1])而非nums[i+1]-nums[i])。代码也偏离ILR：把循环结构修好但丢掉了后缀和递推dp[i][j]=dp[i][j+1]，只做单点转移，[2,3,2]输出1(期望4)、[5,5,5,5]输出1(期望126)。 |
| 3312._Sorted_GCD_Pair_Queries.json | ILR完全错乱不可执行：循环头被写成logic恒真的decision、循环变量x/c/g/q从未赋值（node6即KeyError 'x'）、node17的next指向不存在的node20、end节点node19不可达(16->18->16死循环)、divisors始终为空表。代码自行重构了算法但仍有错：else分支漏加c*(c-1)//2项、筛减和glist/vlist/bisect均未按大小排序，两个示例都抛 IndexError: list index out of range，与FAILURE RESULT一致。 |
| 479._Largest_Palindrome_Product.json | ILR确有错：内层循环标称'j从upper到sqrt(pal)'但node14是j+=1（应递减），且全程无i-=1、外层无法推进，还缺少j本身必须是n位数的守卫——n=3时i=999,pal=999999会接受j=1001(4位数)这一非法因子，输出1260而非123。代码额外偏离ILR：加了while j<=sqrt_pal上界但j从upper递增，upper>sqrt(pal)循环体永不执行，n=2/3都返回-1。 |
| HumanEval-140.json | ILR 确实错（根因成立）：node4 判断 text[i]=='?' 而非空格 ' '，且 node13 每轮把 start/end 都重置为 i+1 使 node6 的 end-start>0 恒假，忠实实现永远输出空串。但代码还额外偏离了 ILR：'?' else 分支用 '_'（ILR 是 '!' + text[i]），非 '?' 分支挪用了 ILR 中不可达 node8 的动作（阈值 2）替代 node6（阈值 0），并未忠实转换。两者都过不了测试，MD 漏了代码偏离。 |
| HumanEval-154.json | ILR 确实错（根因成立）：node8 的 logic 写成了循环守卫 i <= len(a)-l 而非标签中的字符串比较，且整个 j（旋转）循环丢失、node10 的 next 指向不存在的 node12，字面忠实实现只要 len(a)>=len(b) 就立即返回 True。但代码也额外偏离 ILR：自行改成 while+比较的结构，并用 pat[i:i+l]（与 a 同一下标）替代标签里的 pat[j:j+l]，旋转匹配逻辑缺失，'yello'/'ell' 和 'efef'/'fee' 仍失败。两边都错，MD 漏了代码偏离。 |
| HumanEval-36.json | MD 根因判对（ILR 循环无边界条件且非整除分支直接退出到 join，忠实执行 s 恒为空而 KeyError('c')），但漏了代码问题：生成代码补边界时写成 while i<n 先自增后判断，把 n 本身也计入循环，78 可被13整除导致 candidate(78)=3≠2；若边界写对其余断言本可全过。 |
| HumanEval-38.json | MD 根因判对（ILR 子流程不绑定参数→KeyError('s')，encode 子图节点3引用未定义 ctx['group']，且按结构化 arguments decode 只调用一次 encode，decode(encoded)=encode²(原文)≠原文），但漏了代码问题：生成代码把解码逻辑写成 encode_cyclic 调用自身，无限递归崩溃，与 ILR 结构（decode 调 encode）完全不符。 |
| HumanEval-50.json | MD 根因判对（OCR 流程图含 encode/decode 两条分支但 ILR 只收录 encode(+5) 半边，忠实实现作 candidate 是+5编码，0/20），但漏了代码问题：生成代码自行写出 -5 解码逻辑却命名为 encode_shift 并覆盖了测试依赖的 +5 编码，decode_shift 仍是返回 None 的题面桩导致全挂；该函数体若命名为 decode_shift 则 50/50 通过。 |
| HumanEval-67.json | MD 根因判对（ILR 非数字词分支回到 len(words) 检查但从不消费该词，忠实执行死循环），但漏了代码问题：生成代码遇非数字词直接 break，只统计到第一个数字词，19-5=14≠8；正确修法是消费并跳过非数字词继续扫描。 |
| HumanEval-72.json | MD 根因判对（ILR 无输入绑定节点且节点5把 q[i]!=q[j] 为真的分支接到 Return True，不平衡输入返回 True），但漏了代码问题：生成代码镜像该反转，还把循环正常结束（回文）后的返回写成 False（ILR 节点4 false→节点6 是 Return True），连回文样例也全错，6 条断言全挂。 |
| HumanEval-8.json | MD 根因判对（ILR 节点6/7 的 action 自相矛盾：'i' not in ctx 时引用 ctx['i'] 必 KeyError('i')，且循环既不消费元素也不递增 i，忠实实现无法运行），但漏了代码问题：生成代码同时用 numbers[i] 索引（i 递增）和 pop(0) 双重消费元素，任何非空列表都 IndexError。 |
| HumanEval-99.json | MD 根因判对（ILR 节点10 的 action 是 int(num) 向零截断而非标签所称 floor，'-15.5' 得 -15≠-16，忠实实现挂 Test3），但漏了代码问题：生成代码把 rstrip('0') 无条件提到 '.' 判断之前（ILR 中节点4只在含小数点分支内），'10'→'1' 致 Test1 失败且 '0' 会 int('') 崩溃；其负数分支自行补 -1 反而修正了 ILR 的 -15.5 错误。 |
| biweekly-contest-109-visit-array-positions-to-maximize-score.json | 根因确在 ILR：它丢失了 OCR 的 'For i from 1 to N-1' 循环语义，i 从未初始化/自增，条件 i<n-1 漏掉最后一个下标，且 node4 的 false_next(6) 与边 4->10 自相矛盾，任何忠实字面实现都过不了（示例2 [2,4,6,8] 得 12/16 而非 20）。但生成代码还额外偏离 ILR：把顺序执行的 node5、node6（两种转移取 max）改成了按 nums[i] 奇偶二选一，导致示例1 也错，MD 漏掉了这一层。 |
| biweekly-contest-111-count-pairs-whose-sum-is-less-than-target.json | ILR 确实错误：true 分支在 count+=right-left 后又执行 node7 的 right-=1，false 分支经 node8+node9 把 left 加了 2，与正确双指针（true: left+=1 / false: right-=1）不符，忠实转写 89/100 失败且全部与暴力真值不符。但生成代码也未忠实转写 ILR——它整体丢弃了 node 9 的无条件 left+=1（true 分支少了 left+=1，false 分支只加 1 而非加 2），与忠实转写在 35 组测试上输出不同，且自己也不正确（示例1 输出 0 而期望 3），MD 漏掉了代码层问题。 |
| biweekly-contest-111-number-of-beautiful-integers-in-the-range.json | ILR 多处硬伤：tight 时 bound=0（应为 num[pos] 的数字值）、i==0 且无首位时反而置 new_has_first=True（逻辑反了）、奇数位经 node28+29 抵消成 even_num-1、node30/31 对 dp 结果重复累加、且从未给出 dp(0,True,False,0,0) 的初始调用，忠实转写 66/100 失败。生成代码也偏离 ILR：删掉了 node29 的覆盖和 node31 的重复累加，导致几乎恒返回 0（62/100 失败，示例1 输出 0 而期望 2），MD 漏了代码层偏离。 |
| biweekly-contest-114-minimum-operations-to-collect-elements.json | ILR 丢失了 OCR 的 node5『j = abs(nums[i]) - 1』，导致 node4 引用的 j 从未定义，按字面执行直接 NameError；且循环无 i 自减、node6『循环结束』却指回循环头 node3。即使最小修复（补回 j、补 i-=1），其算法对重复元素也会重复计数（标记的是位置 i 而非值槽），仍有 11/100 失败。生成代码则把 j<k 擅自解释成 i<k（下标与 k 比较）并把继续扫描改成 break，与修复版 ILR 在 72/100 组上输出不同，示例1 输出 1 而期望 4，MD 漏了代码层严重偏离。 |
| biweekly-contest-114-split-array-into-maximum-number-of-subarrays.json | ILR 丢失了 OCR 的前段循环（m = m & nums[i] 求全数组 AND），m 恒为 33554431，node3 的 m==0 判断恒假只能走 false_next=19，而 ILR 节点表里根本没有 19 号节点（Return 1 成了孤儿节点 12），且计数循环内没有任何 i 自增节点，字面实现恒返回 1（76/100 失败）。生成代码则把计数循环复制到了 else 分支且从不返回 1，还把 node8 的 false 分支改成 i+=1，实际恒返回 0（100/100 全错，示例1 得 0 期望 3，示例2 得 0 期望 1），MD 漏了代码层偏离。 |
| biweekly-contest-116-subarrays-distinct-element-sum-of-squares-i.json | ILR 自身不可执行：任何节点都没有 i+=1（外层死循环），node3 的 false_next=6 指向不存在的节点（OCR 的 Output: result 丢失）；且 node8 在 j 循环外多加一次 len(s)^2，即使补上 i 自增也每组测试都错（100/100）。生成代码则把 node9 的『每个子数组统计一次』从内层循环里整个删掉，只在每轮 i 结束后算一次后缀去重数的平方（95/100 失败），示例1 输出 9 而期望 15，与修复版 ILR 输出 100 组全不同，MD 漏了代码层偏离。 |
| biweekly-contest-118-minimum-number-of-coins-for-fruits.json | MD 只说对一半：ILR 确实有错——节点12 和节点14 都执行 append（12 的 next 指向 14），else 路径会把 (pos,new) 入队两次，污染单调队列导致答案偏小。但代码还额外偏离了 ILR：它把 while 弹尾+append 挪进了 if i==temp[0][0] 的 else 分支，而 ILR/流程图两条分支都必须经过弹尾和节点14 的 append，导致弹出队首后不再入队、temp 变空后下一次迭代崩溃。 |
| biweekly-contest-120-count-the-number-of-incremovable-subarrays-i.json | MD 只说对一半。ILR 确实错得离谱：节点10（k 不在删除区间）与节点13（违反递增）的动作都是 ans+=1 且随后置 ok=False，节点11 更新 lst 后回到判断节点9 立即判假，ans 在 k 循环内被反复累加且无任何 i/j/k 递增节点，忠实执行既死循环也得不到正确计数。但代码也额外偏离：它把 k 在删除区间内/外两个分支写成完全相同的检查（从不跳过 nums[i..j]），等价于只检查整个数组是否严格递增，因此在非整体递增数组上计数为 0。 |
| biweekly-contest-120-count-the-number-of-incremovable-subarrays-ii.json | MD 只说对一半。ILR 的节点7 把 break 误写成赋值 i=n-1，使节点8 的'循环正常完成'判断恒真，节点10-17（两指针主体）不可达，忠实执行对任何输入都返回 (n-2)*(n-1)//2（仅整体递增时碰巧正确）；节点12 同样误写 j=0，节点15↔16 还构成死循环环。代码则额外偏离 ILR：完全省略了节点8 的提前返回分支，带着被污染的 i=n-1、j=0 去跑两指针，且 r 可越过 n 导致 res 出现负数。 |
| weekly-contest-352-longest-even-odd-subarray-with-threshold.json | MD 只说对一半。ILR 确实错得严重：节点4 的 action 是悬空的 'for ...:'（无循环体，语法错误），节点9 的 action 是裸 continue（语句环境中不可执行），ans 更新节点8 只挂在违规路径 6→8 上且其后直接进 END（节点10），节点7-false 也直达 END，结构无法表达正确算法。代码额外偏离：它把 ans 更新挪到 else 分支（仅在窗口当前合法时更新），left 刚跳到新起点的那次迭代不更新 ans，与 ILR 字面结构（true 分支更新 ans）和正确解法（无条件更新）都不同。 |
| weekly-contest-352-sum-of-imbalance-numbers-of-all-subarrays.json | MD 只说对一半。ILR 的错误：内层循环变量 j 在全图无任何初始化和递增节点（忠实执行死循环），且 OCR 流程图的 'For j from i to n-1' 被丢成 'Is j < n?'，起点 i 丢失。代码额外偏离：节点9/11 的 r += temp 原本在 j 循环内每个 j 累加一次，代码把它挪到 j 循环结束后每个 i 只加一次，导致结果完全错误。 |
| weekly-contest-353-apply-operations-to-make-all-array-elements-equal-to-zero.json | MD 只说对一半。ILR 的错误：节点14（result=True）在图中没有任何入边、成为孤点，循环正常走完到节点16 时 ctx['result'] 从未被赋值，忠实执行会 KeyError，即 ILR 永远无法返回 True。代码额外偏离：ILR 节点13（11→13→12）要求在 n>h 分支设置 h=n 后再继续，代码却把 h=n 挪到 n==h 的 else 分支（在该分支是无效操作），n>h 分支漏掉了这一关键更新，导致 h 持续偏低、大量本应成功的用例误判 False。 |
| weekly-contest-364-maximum-odd-binary-number.json | ILR 确实错误（c==1 且 len(s)>1 时在 node3/5/7/9/10 间死循环，s='1' 时输出空串）；但代码还额外自行偏离：把 c==1 分支错误简化为直接返回 '1'，丢掉'前面补 0'的逻辑，在所有 c==1 且长度>1 的用例上失败——这部分是代码生成阶段的独立错误，MD 只归因 ILR 不完整。 |
| weekly-contest-370-find-champion-i.json | ILR 确实错误（j 未定义且从不递增 → KeyError，node8 break 指回 node4 死循环，且检查恒为 0 的 grid[i][i]）；但代码还自行把判断改成 grid[i][j]==1 就置 isChampion=False（方向反了，等价于选最弱队伍），即使 ILR 正确这段代码也过不了，属于代码生成阶段的独立错误，MD 只归因 ILR 不完整。 |
| weekly-contest-371-maximum-strong-pair-xor-ii.json | MD 根因(ILR错)正确但不完整。ILR 把 node8 的真假分支接反（p in pref 时反而执行覆盖式 Set，p 不在时执行对普通 dict 的 max(pref2[p],a) → KeyError），node11 还是 node10 的重复；忠实实现必然 KeyError，即使宽容用 defaultdict 也在 [10,100] 上得 110(期望0)。代码偏离 ILR 修正了分支，但漏掉了 ILR node4 的 res <<= 1，导致只置最低位。 |
| weekly-contest-371-minimum-operations-to-maximize-last-elements-in-arrays.json | MD 根因(ILR错)正确但不完整。ILR 把正确的 DP 双转移写成 if/else 二选一（node8 真时只做 node9、假时只做 node10；node11 同理），且控制流断裂（node13 直接到 end，循环体内无 i-=1，i 未初始化）：修好控制流后 [1,2,7]/[4,5,3] 仍得 2(期望1)。代码修复了循环结构但自行把 node4 的全量初始化改成只初始化 dp[0]（且随即被覆盖），丢掉了基例 dp[n-1]=(0,1)，导致几乎全部输出 -1。 |
| weekly-contest-373-count-beautiful-substrings-i.json | MD 根因(ILR错)正确但不完整。ILR 中元音/辅音分支经 node7、node8 后对每个字符同时给 cnt_0 和 cnt_1 各 +1（两个计数都等于位置数），第二阶段 v1=cnt_0[i]-cnt_0[i] 恒为 0 且 node11 每次把 j 重置为 i+1 → 忠实执行死循环。代码偏离 ILR 重写了循环，但前缀计数没有滚动继承（辅音处 cnt_0[i+1] 保持 0 而非等于 cnt_0[i]），独立地错误，81 个断言失败。 |
| weekly-contest-374-minimum-number-of-coins-to-be-added.json | MD 根因(ILR错)正确但不完整。ILR 混乱：node7 后 node8 重复 ans+=1、node12 后 node9 重复 max_sofar 更新、可用硬币路径(node12)做的是加虚拟硬币而非 max_sofar+=coin、先 pop 再判断导致硬币被丢弃、node15 用过期 coin 循环：忠实实现示例1 就得 3(期望2)。代码修复了部分算术，但继承了“先 pop 后判断”的丢币结构，且把 coins 用尽后的补充阶段(node11/15)整个删掉，[1,1,1],20 → 0(期望3)。 |
| weekly-contest-375-count-subarrays-where-max-element-appears-at-least-k-times.json | MD 根因(ILR错)正确但不完整。ILR 丢失了收缩循环回边（OCR 原图有 11→9，ILR 把 node11 接到 node3），每个 val 最多收缩一步即 res+=left：忠实实现 [1,3,2,3,3],k=2 → 3(期望6)。代码自造了 while nums[left]==maxVal 的收缩守卫（ILR node6 的 count>=k 才是正确循环条件），遇非 max 元素立即停止收缩，示例1 直接得 0，另有 95 个断言失败。 |
| weekly-contest-376-apply-operations-to-maximize-frequency-score.json | MD 根因(ILR错)正确但不完整。ILR 在排序(node3)之前就计算 sum_vals(node2)，前缀和基于未排序数组而 nums[mid] 用排序后数组，两者不一致；循环条件 right < len(sum_vals)-1 少算最后一个元素；图中无 right 的初始化与递增节点。忠实实现 7 个样例错 5 个（示例2 得 4 期望 3）。代码另有独立致命伤：把 ctx['arg0'] 写成 self.arg0（AttributeError，即记录的失败原因），还漏掉了 node3 的 nums.sort()。 |

## 5. 值得注意的细节

- **MATH/2514（Count Anagrams）**：生成代码数学上完全正确、全部断言通过，失败原因是超时（'a'*100000 一项约 8.3 秒）；MD 标 "ILR Error" 不成立（ILR 图虽有接线错误但未被代码继承）。此类"超时"失败在 MD 中一律未与逻辑错误区分。
- **MATH/780（Reaching Points）**：ILR 忠实转写 9/9 测试通过；但 ILR 存在未被测试暴露的潜在缺陷（ty==sy 时会误判，如 (2,2,3,2)），若换测试集可能暴露——按当前测试口径归为代码错误。
- **qwen3-vl-8b 的代码生成特点**（与 gpt-4o-mini 明显不同）：大量出现"半修复"行为——代码重建/修补了 ILR 损坏的控制流（死循环、缺自增），却保留或新引入其他缺陷；45/111 失败属于此类 both。纯代码错误仅 6 题（5%），远低于 gpt-4o-mini（7/81，9%）。
- ILR 缺陷高度集中于**循环机制丢失/错译**（循环变量无初始化/无递增、break 译成赋值、回边指错、悬空 next 引用）、**分支接反**、**节点动作与标签矛盾**、**丢节点/丢分支**。
