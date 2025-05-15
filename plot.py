import matplotlib.pyplot as plt
import numpy as np

# Acevedo-20示例数据
# client_ids = ['Client1', 'Client1','Client1', 'Client1', 'Client1','Client1','Client1', 'Client1', 'Client1','Client1','Client2', 'Client2','Client2','Client2','Client2', 'Client2','Client2','Client2','Client2', 'Client2', 'Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4',
#               'Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6']
# category_ids = ['1', '2', '3','4','5','6','7','8','9','10', '1', '2', '3','4','5','6','7','8','9','10', '1', '2', '3','4','5','6','7','8','9','10','1', '2', '3','4','5','6','7','8','9','10','1', '2', '3','4','5','6','7','8','9','10','1', '2', '3','4','5','6','7','8','9','10']
# quantities = [210, 244, 102,242, 629,5,196,320, 46,145,   189,245,103,261,113,191,112, 62,589,15,       178,269,101,250, 212,187, 212, 110,29 ,88,  190,272,82,254,971,196,112, 516,147,414,
#               207,275,85,231,568, 105,176,308,313,88,  0,0,0,0,0,225,4,0,12,221]
#Matek-19数据局分布
# client_ids=['Client1', 'Client1','Client1', 'Client1', 'Client1','Client1','Client1', 'Client1', 'Client1','Client1','Client1','Client1', 'Client1', 'Client2', 'Client2','Client2','Client2','Client2', 'Client2','Client2','Client2','Client2', 'Client2','Client2','Client2','Client2', 'Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4',
#               'Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6',
#             'Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7']
#
# category_ids=['1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13']
# quantities=[16,15,291,10,79,5,765,5,1421,1,28,1516,1,   12,14,295,11,18,839,5,1,2521,0,18,167,0,   12,24,287,9,29,346,11,1,44,0,1,128,1,    13,16,288,15,131,322,3,4,1413,0,3,513,1, 10,18,287,17, 82,219,1, 1,514,6,6,411,3,  0,0,0,0,0, 123,8,0,874,1,0,106,2,
#             0,0,0,0,0,0,0,0,0,0,0,308,4]
##HLwbc
client_ids=['Client1', 'Client1','Client1', 'Client1', 'Client1','Client1','Client1', 'Client1', 'Client2', 'Client2','Client2','Client2', 'Client2','Client2','Client2','Client2', 'Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4',
            'Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6'
           ]

category_ids=['1', '2', '3','4','5','6','7','8','1', '2', '3','4','5','6','7','8','1', '2', '3','4','5','6','7','8','1', '2', '3','4','5','6','7','8','1', '2', '3','4','5','6','7','8','1', '2', '3','4','5','6','7','8']
quantities=[166,13,78,166,64,95,48,446,    164,12,84,163,33,965,27,121,   165,14,90,154,94,363,79,504,    158,24,69,172,1269,155,1021,958,     164,16,87,156,507,981,409,183,   0,0,0,0,60,1,48,633]
# task:0,client data num distribution:[423, 423, 423, 423, 423]
# task:0,client:0,calss num:Counter({3: 166, 0: 166, 2: 78, 1: 13})
# task:0,client:1,calss num:Counter({0: 164, 3: 163, 2: 84, 1: 12})
# task:0,client:2,calss num:Counter({0: 165, 3: 154, 2: 90, 1: 14})
# task:0,client:3,calss num:Counter({3: 172, 0: 158, 2: 69, 1: 24})
# task:0,client:4,calss num:Counter({0: 164, 3: 156, 2: 87, 1: 16})
# task:1,client data num distribution:[159, 998, 457, 1424, 1488, 61]
# task:1,client:0,calss num:Counter({5: 95, 4: 64})
# task:1,client:1,calss num:Counter({5: 965, 4: 33})
# task:1,client:2,calss num:Counter({5: 363, 4: 94})
# task:1,client:3,calss num:Counter({4: 1269, 5: 155})
# task:1,client:4,calss num:Counter({5: 981, 4: 507})
# task:1,client:5,calss num:Counter({4: 60, 5: 1})
# task:2,client data num distribution:[494, 148, 583, 1979, 592, 681]
# task:2,client:0,calss num:Counter({7: 446, 6: 48})
# task:2,client:1,calss num:Counter({7: 121, 6: 27})
# task:2,client:2,calss num:Counter({7: 504, 6: 79})
# task:2,client:3,calss num:Counter({6: 1021, 7: 958})
# task:2,client:4,calss num:Counter({6: 409, 7: 183})
# task:2,client:5,calss num:Counter({7: 633, 6: 48})



###CDIL
# client_ids=['Client1', 'Client1','Client1', 'Client1', 'Client1','Client1','Client1', 'Client1', 'Client1','Client1','Client1','Client1', 'Client1', 'Client2', 'Client2','Client2','Client2','Client2', 'Client2','Client2','Client2','Client2', 'Client2','Client2','Client2','Client2', 'Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client3','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4','Client4',
#               'Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client5','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6','Client6',
#             'Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7','Client7']
# category_ids=['1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13','1', '2', '3','4','5','6','7','8','9','10','11','12','13']
# quantities=[16,15,291,10,173,249,2,7,101,1,15,272,1,]
# task:0,client:0,calss num:Counter({2: 291, 0: 16, 1: 15, 3: 10})
# task:0,client:1,calss num:Counter({2: 295, 1: 14, 0: 12, 3: 11})
# task:0,client:2,calss num:Counter({2: 287, 1: 24, 0: 12, 3: 9})
# task:0,client:3,calss num:Counter({2: 288, 1: 16, 3: 15, 0: 13})
# task:0,client:4,calss num:Counter({2: 287, 1: 18, 3: 17, 0: 10})
# task:1,client:[424, 136, 522, 952, 952]
# task:1,client:0,calss num:Counter({5: 249, 4: 173, 6: 2})
# task:1,client:1,calss num:Counter({4: 72, 5: 62, 6: 2})
# task:1,client:2,calss num:Counter({5: 513, 4: 9})
# task:1,client:3,calss num:Counter({5: 861, 4: 66, 6: 25})
# task:1,client:4,calss num:Counter({5: 929, 4: 19, 6: 4})
# task:2,client:[109, 504, 44, 2192, 3958]
# task:2,client:0,calss num:Counter({8: 101, 7: 7, 9: 1})
# task:2,client:1,calss num:Counter({8: 501, 7: 2, 9: 1})
# task:2,client:2,calss num:Counter({8: 43, 9: 1})
# task:2,client:3,calss num:Counter({8: 2185, 9: 4, 7: 3})
# task:2,client:4,calss num:Counter({8: 3957, 9: 1})
# task:3,client:[288, 318, 44, 345, 2017, 205]
# task:3,client:0,calss num:Counter({11: 272, 10: 15, 12: 1})
# task:3,client:1,calss num:Counter({11: 308, 12: 6, 10: 4})
# task:3,client:2,calss num:Counter({11: 37, 10: 7})
# task:3,client:3,calss num:Counter({11: 324, 10: 18, 12: 3})
# task:3,client:4,calss num:Counter({11: 2004, 10: 12, 12: 1})
# task:3,client:5,calss num:Counter({11: 204, 12: 1})
# task:4,client:[324, 1200, 790, 137, 1943, 262]
# task:4,client:0,calss num:Counter({2: 161, 1: 85, 0: 70, 3: 8})
# task:4,client:1,calss num:Counter({1: 401, 2: 379, 3: 244, 0: 176})
# task:4,client:2,calss num:Counter({3: 362, 2: 241, 1: 185, 0: 2})
# task:4,client:3,calss num:Counter({0: 111, 2: 22, 1: 4})
# task:4,client:4,calss num:Counter({1: 631, 0: 605, 3: 403, 2: 304})
# task:4,client:5,calss num:Counter({3: 223, 2: 29, 0: 10})
# task:5,client:[830, 416, 611, 1279, 849, 229]
# task:5,client:0,calss num:Counter({4: 629, 7: 196, 6: 5})
# task:5,client:1,calss num:Counter({6: 191, 4: 113, 7: 112})
# task:5,client:2,calss num:Counter({4: 212, 7: 212, 6: 187})
# task:5,client:3,calss num:Counter({4: 971, 6: 196, 7: 112})
# task:5,client:4,calss num:Counter({4: 568, 7: 176, 6: 105})
# task:5,client:5,calss num:Counter({6: 225, 7: 4})
# task:6,client:[751, 670, 421, 328, 310, 211, 69]
# task:6,client:0,calss num:Counter({8: 674, 10: 64, 11: 13})
# task:6,client:1,calss num:Counter({8: 396, 11: 271, 10: 3})
# task:6,client:2,calss num:Counter({11: 261, 8: 88, 10: 72})
# task:6,client:3,calss num:Counter({11: 232, 10: 62, 8: 34})
# task:6,client:4,calss num:Counter({11: 141, 8: 123, 10: 46})
# task:6,client:6,calss num:Counter({11: 37, 10: 31, 8: 1})
# task:6,client:5,calss num:Counter({10: 195, 11: 16})
# task:7,client:[576, 315, 273, 360, 884, 162, 367]
# task:7,client:0,calss num:Counter({0: 311, 2: 199, 3: 66})
# task:7,client:1,calss num:Counter({2: 274, 3: 36, 1: 3, 0: 2})
# task:7,client:2,calss num:Counter({0: 195, 3: 65, 1: 7, 2: 6})
# task:7,client:3,calss num:Counter({0: 179, 2: 150, 1: 31})
# task:7,client:4,calss num:Counter({2: 831, 0: 28, 3: 19, 1: 6})
# task:7,client:5,calss num:Counter({0: 88, 3: 65, 2: 9})
# task:7,client:6,calss num:Counter({2: 163, 3: 157, 1: 32, 0: 15})
# task:8,client:[860, 811, 741, 655, 922, 1410, 837]
# task:8,client:0,calss num:Counter({4: 412, 11: 368, 8: 80})
# task:8,client:1,calss num:Counter({8: 323, 4: 248, 11: 240})
# task:8,client:2,calss num:Counter({8: 432, 11: 254, 4: 55})
# task:8,client:3,calss num:Counter({8: 612, 11: 23, 4: 20})
# task:8,client:4,calss num:Counter({8: 535, 11: 310, 4: 77})
# task:8,client:6,calss num:Counter({8: 564, 11: 272, 4: 1})
# task:8,client:5,calss num:Counter({11: 1396, 8: 14})

###DIL

# 使用字典汇总数量
data = {}
for client_id, category_id, quantity in zip(client_ids, category_ids, quantities):
    key = (client_id, category_id)
    if key in data:
        data[key] += quantity
    else:
        data[key] = quantity

# 分离数据
clients, categories = zip(*data.keys())
sizes =list(data.values())

# 将类别转换为数字，以便绘图
# 将类别转换为整数并排序
unique_categories = sorted(set(map(int, categories)))
category_indices = [unique_categories.index(int(cat)) for cat in categories]

# 创建图形
plt.figure(figsize=(5, 6))

# 为每个类别分配一个颜色
#colors = plt.cm.viridis(np.linspace(0, 1, len(unique_categories)))
# 为每个类别分配一个颜色
# 为每个类别分配一个颜色
color_map = plt.cm.get_cmap('tab10', len(unique_categories))  # 使用 tab10 colormap
category_colors = {i: color_map(i) for i in unique_categories}  # 使用整数作为键


# 使用散点图
scatter = plt.scatter(clients, category_indices, s=[size * 2 for size in sizes],
                      c='red', alpha=1, cmap='viridis', edgecolor='none')
#[category_colors[int(cat)] for cat in categories]
#连接相同客户端的散点
for client in set(clients):
    indices = [i for i, c in enumerate(clients) if c == client]
    plt.plot([client] * len(indices), [category_indices[i] for i in indices], color='black', linestyle='-', marker='o',markersize=0.5,linewidth=0.5)

# # 连接同一横轴上的散点
# for category in set(category_indices):
#     indices = [i for i, c in enumerate(category_indices) if c == category]
#     plt.plot([clients[i] for i in indices], [category_indices[i] for i in indices],
#              color='black', linestyle='-', marker='o', markersize=0.5, linewidth=0.5)

#添加颜色条
# cbar = plt.colorbar(scatter)
# cbar.set_label('Sample Quantity')


# 添加标题和标签
plt.title('Client Sample Distribution')
plt.xlabel('Client ID')
plt.ylabel('Class ID')

# 设置 y 轴刻度为类别
plt.yticks(ticks=range(len(unique_categories)), labels=unique_categories)
# 设置 y 轴范围，确保从小到大
plt.ylim(-0.5, len(unique_categories) - 0.5)
# 显示图形
plt.xticks(rotation=35)

plt.tight_layout(w_pad=2.0, h_pad=3.0)
plt.show()