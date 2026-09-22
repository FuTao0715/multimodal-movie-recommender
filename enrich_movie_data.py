#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
enrich_movie_data.py — TMDB 电影数据补全脚本（并发加速版）
===============================================================================
功能说明：
    读取 ml-latest-small 数据集中的 links.csv 和 movies.csv，
    通过 TMDB API 并发获取每部电影的剧情简介 (overview) 和海报图片 (poster)，
    最终输出一个包含完整信息的 movies_enriched.csv 文件。

性能说明：
    使用 ThreadPoolExecutor 实现并发请求，默认 10 线程，
    配合令牌桶速率限制，比单线程版快约 10 倍。

使用方法：
    python enrich_movie_data.py

输出：
    - movies_enriched.csv  （补全后的电影数据表）
    - ./posters/ 目录     （下载的电影海报图片）
===============================================================================
"""

# ============================================================================
# [配置区] 所有可调参数集中在此处，方便修改
# ============================================================================

# TMDB API 配置
TMDB_API_KEY = "8ac6862bd7bb333f8b040f132b5bc196"       # 你的 TMDB API v3 Key
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie/"    # Movie Details 接口基础 URL
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"  # 海报图片基础 URL（w500 尺寸）
TMDB_LANGUAGE = "en-US"                                  # 请求语言

# 文件路径配置
LINKS_CSV = "ml-latest-small/links.csv"                  # 输入：映射文件
MOVIES_CSV = "ml-latest-small/movies.csv"                # 输入：电影标题文件
OUTPUT_CSV = "movies_enriched.csv"                       # 输出：补全后的 CSV
POSTER_DIR = "posters"                                   # 输出：海报保存目录

# 请求控制配置
CONCURRENT_WORKERS = 10     # 并发线程数（太大容易触发 429，建议 8~15）
REQUESTS_PER_SECOND = 5.0   # 全局每秒最大请求数（TMDB 免费 Key 建议 ≤5）
REQUEST_TIMEOUT = 15        # 单次请求超时时间（秒）
MAX_RETRIES = 3             # 单部电影最大重试次数
RATE_LIMIT_RETRY_WAIT = 1.0 # 触发 429 后的冷却等待（秒）

# 运行控制配置
ENABLE_IMAGE_DOWNLOAD = True # 是否下载海报图片（设为 False 则只获取文字信息）
MAX_MOVIES = None            # 最大处理数量（None = 全部处理，设数字可限制测试量）


# ============================================================================
# [模块 0] 基础库导入（内置库，无需安装）
# ============================================================================
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count


# ============================================================================
# [模块 1] 依赖检查与自动安装
# ============================================================================

def check_and_install_dependencies():
    """
    检测 pandas 和 requests 是否已安装；
    如果未安装，则尝试通过 pip 自动安装。
    """
    import subprocess

    missing_packages = []

    for package in ["pandas", "requests"]:
        try:
            __import__(package)
            print(f"[OK] 依赖包 '{package}' 已安装。")
        except ImportError:
            print(f"[MISS] 依赖包 '{package}' 未安装，将自动安装...")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n正在安装缺失的依赖包: {', '.join(missing_packages)} ...")
        for pkg in missing_packages:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"[OK] '{pkg}' 安装成功！")
            except subprocess.CalledProcessError:
                print(f"[ERROR] '{pkg}' 自动安装失败，请手动执行: pip install {pkg}")
                sys.exit(1)
        print("所有依赖安装完毕，重新导入模块...\n")


# ============================================================================
# [模块 2] 数据读取与映射
# ============================================================================

def load_data():
    """
    读取 links.csv 和 movies.csv，合并为待查询列表。
    返回:
        movie_list: list of dict，每个元素包含 movieId, title, tmdbId
    """
    import pandas as pd

    print("\n" + "=" * 60)
    print(">>> 步骤 1：读取本地数据文件")
    print("=" * 60)

    links_df = pd.read_csv(LINKS_CSV)
    print(f"  从 '{LINKS_CSV}' 读取到 {len(links_df)} 条映射记录")

    movies_df = pd.read_csv(MOVIES_CSV)
    print(f"  从 '{MOVIES_CSV}' 读取到 {len(movies_df)} 部电影标题")

    merged_df = pd.merge(movies_df, links_df, on="movieId", how="inner")
    print(f"  合并后共 {len(merged_df)} 条记录")

    valid_df = merged_df[merged_df["tmdbId"].notna() & (merged_df["tmdbId"] > 0)].copy()
    valid_df["tmdbId"] = valid_df["tmdbId"].astype(int)
    print(f"  其中 tmdbId 有效的记录: {len(valid_df)} 条")

    if MAX_MOVIES is not None:
        valid_df = valid_df.head(MAX_MOVIES)
        print(f"  [测试模式] 仅处理前 {MAX_MOVIES} 部电影")

    movie_list = valid_df[["movieId", "title", "tmdbId"]].to_dict("records")
    print(f"  最终待处理电影数量: {len(movie_list)} 部")
    return movie_list


# ============================================================================
# [模块 3] 全局限速器（线程安全）
# ============================================================================

class RateLimiter:
    """
    线程安全的令牌桶限速器。
    确保全局每秒请求数不超过设定值，防止触发 TMDB 429。
    """
    def __init__(self, rate_per_second):
        self._min_interval = 1.0 / rate_per_second  # 两次请求之间的最小间隔
        self._lock = threading.Lock()
        self._last_request_time = 0.0

    def acquire(self):
        """获取请求许可，必要时阻塞等待。"""
        with self._lock:
            now = time.time()
            wait = self._last_request_time + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
                self._last_request_time = time.time()
            else:
                self._last_request_time = now

    def cooldown(self, seconds):
        """触发限流后的强制冷却（如收到 429）。"""
        with self._lock:
            self._last_request_time = time.time() + seconds


# ============================================================================
# [模块 4] 进度计数器（线程安全）
# ============================================================================

class ProgressCounter:
    """线程安全的进度计数器。"""
    def __init__(self, total):
        self.total = total
        self.done = 0
        self.success_api = 0   # API 请求成功
        self.success_img = 0   # 图片下载成功
        self.fail_api = 0      # API 请求失败
        self._lock = threading.Lock()

    def update(self, api_ok=False, img_ok=False):
        """更新计数（调用一次代表完成一部电影的处理）。"""
        with self._lock:
            self.done += 1
            if api_ok:
                self.success_api += 1
            else:
                self.fail_api += 1
            if img_ok:
                self.success_img += 1

    def snapshot(self):
        """获取当前进度的瞬时快照。"""
        with self._lock:
            return (self.done, self.success_api, self.success_img, self.fail_api)


# ============================================================================
# [模块 5] API 请求核心逻辑
# ============================================================================

def fetch_movie_details(tmdb_id, session, rate_limiter):
    """
    向 TMDB API 请求单部电影的详细信息（带重试与限速）。
    参数:
        tmdb_id       : TMDB 电影 ID
        session       : requests.Session 对象（线程本地）
        rate_limiter  : RateLimiter 实例
    返回:
        (poster_path, overview) 元组；失败时返回 (None, None)
    """
    import requests as req_lib

    url = f"{TMDB_BASE_URL}{tmdb_id}"
    params = {"api_key": TMDB_API_KEY, "language": TMDB_LANGUAGE}

    for attempt in range(1, MAX_RETRIES + 1):
        # ---- 全局限速 ----
        rate_limiter.acquire()

        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                data = response.json()
                poster_path = data.get("poster_path")
                overview = data.get("overview", "")
                return (poster_path, overview)

            elif response.status_code == 404:
                return (None, None)  # 不存在，不重试

            elif response.status_code == 429:
                # 触发频率限制 → 全局冷却，然后重试
                rate_limiter.cooldown(RATE_LIMIT_RETRY_WAIT)
                continue

            else:
                time.sleep(0.5)
                continue

        except req_lib.exceptions.Timeout:
            time.sleep(0.5)
        except req_lib.exceptions.ConnectionError:
            time.sleep(1.0)
        except Exception:
            return (None, None)

    return (None, None)


# ============================================================================
# [模块 6] 资源下载与保存
# ============================================================================

def download_poster(movie_id, poster_path, session):
    """
    下载电影海报图片到本地 ./posters/ 目录。
    返回:
        本地图片路径（字符串）；失败时返回空字符串
    """
    if not poster_path:
        return ""

    poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
    local_filename = f"{movie_id}.jpg"
    local_path = os.path.join(POSTER_DIR, local_filename)

    # 已存在则跳过
    if os.path.exists(local_path):
        return local_path.replace("\\", "/")

    try:
        response = session.get(poster_url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            return local_path.replace("\\", "/")
        return ""
    except Exception:
        return ""


# ============================================================================
# [模块 7] 单部电影的完整处理流程（供线程池调用）
# ============================================================================

def process_one_movie(movie, rate_limiter, index):
    """
    处理单部电影的完整流程：API 查询 → 海报下载。
    此函数由线程池中的工作线程调用。
    参数:
        movie        : dict，包含 movieId, title, tmdbId
        rate_limiter : RateLimiter 实例
        index        : 原始序号（用于结果排序）
    返回:
        (index, result_dict) 元组
    """
    import requests as req_lib

    movie_id = movie["movieId"]
    title = movie["title"]
    tmdb_id = movie["tmdbId"]

    # 每个线程使用独立的 Session（线程安全）
    session = req_lib.Session()
    session.headers.update({
        "User-Agent": "MovieDataEnricher/2.0 (Educational Project)"
    })

    api_ok = False
    img_ok = False

    try:
        # ---- API 请求 ----
        poster_path, overview = fetch_movie_details(tmdb_id, session, rate_limiter)

        if overview is None:
            overview = ""
        else:
            api_ok = True

        # ---- 海报下载 ----
        poster_local_path = ""
        if ENABLE_IMAGE_DOWNLOAD and poster_path:
            poster_local_path = download_poster(movie_id, poster_path, session)
            if poster_local_path:
                img_ok = True

        result = {
            "movieId": movie_id,
            "title": title,
            "poster_local_path": poster_local_path,
            "overview": overview,
            "_api_ok": api_ok,
            "_img_ok": img_ok,
        }
        return (index, result)

    finally:
        session.close()


# ============================================================================
# [模块 8] 进度显示线程
# ============================================================================

def progress_reporter(progress_counter, stop_event, total):
    """
    每 2 秒在控制台打印一次总体进度。
    运行在独立线程中，收到 stop_event 信号后退出。
    """
    start = time.time()
    while not stop_event.is_set():
        stop_event.wait(2.0)  # 每 2 秒刷新一次
        if stop_event.is_set():
            break
        done, ok, img, fail = progress_counter.snapshot()
        elapsed = time.time() - start
        pct = done / total * 100 if total else 0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"\r  已处理 {done}/{total} ({pct:.1f}%) | "
              f"API成功:{ok} 海报:{img} 失败:{fail} | "
              f"速度:{rate:.1f}部/秒 | 预计剩余:{eta:.0f}秒  ", end="", flush=True)


# ============================================================================
# [模块 9] 主执行流程
# ============================================================================

def main():
    """主函数：串联所有处理步骤（并发版）。"""
    import pandas as pd

    print("=" * 60)
    print("  TMDB 电影数据补全脚本（并发加速版）")
    print(f"  并发线程: {CONCURRENT_WORKERS} | 速率限制: {REQUESTS_PER_SECOND} req/s")
    print("=" * 60)

    # ---- 9.1 依赖检查 ----
    check_and_install_dependencies()

    # ---- 9.2 创建海报保存目录 ----
    if ENABLE_IMAGE_DOWNLOAD and not os.path.exists(POSTER_DIR):
        os.makedirs(POSTER_DIR)
        print(f"[INFO] 已创建海报目录: '{POSTER_DIR}/'")

    # ---- 9.3 加载数据 ----
    movie_list = load_data()
    total = len(movie_list)

    if total == 0:
        print("\n[ERROR] 没有可处理的电影，请检查 links.csv 中的数据。")
        return

    # ---- 9.4 初始化限速器 & 进度计数器 ----
    rate_limiter = RateLimiter(REQUESTS_PER_SECOND)
    progress_counter = ProgressCounter(total)

    # ---- 9.5 启动进度显示线程 ----
    stop_event = threading.Event()
    reporter_thread = threading.Thread(
        target=progress_reporter,
        args=(progress_counter, stop_event, total),
        daemon=True
    )

    # ---- 9.6 并发处理 ----
    print("\n" + "=" * 60)
    print(">>> 步骤 2：并发请求 TMDB API 并下载资源")
    print("=" * 60)

    start_time = time.time()
    reporter_thread.start()

    # 用于按原始顺序排列结果
    results_by_index = {}

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            # 提交所有任务
            futures = {
                executor.submit(process_one_movie, movie, rate_limiter, idx): idx
                for idx, movie in enumerate(movie_list)
            }

            # 收集结果（as_completed 保证拿到一个就处理一个）
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results_by_index[idx] = result

                    # 更新进度计数
                    progress_counter.update(
                        api_ok=result.get("_api_ok", False),
                        img_ok=result.get("_img_ok", False),
                    )
                except Exception as e:
                    # 极端情况：某任务抛出未捕获异常
                    idx = futures[future]
                    progress_counter.update(api_ok=False, img_ok=False)
                    results_by_index[idx] = {
                        "movieId": movie_list[idx]["movieId"],
                        "title": movie_list[idx]["title"],
                        "poster_local_path": "",
                        "overview": "",
                        "_api_ok": False,
                        "_img_ok": False,
                    }

    finally:
        # 停止进度显示线程
        stop_event.set()
        reporter_thread.join(timeout=3)

    # 换行（清除进度行）
    print()

    # ---- 9.7 按原始顺序重建结果列表 ----
    enriched_data = []
    for i in range(total):
        r = results_by_index.get(i)
        if r is None:
            # 防御：如果某条记录丢失（理论上不会发生），补一条空记录
            m = movie_list[i]
            enriched_data.append({
                "movieId": m["movieId"],
                "title": m["title"],
                "poster_local_path": "",
                "overview": "",
            })
        else:
            enriched_data.append({
                "movieId": r["movieId"],
                "title": r["title"],
                "poster_local_path": r["poster_local_path"],
                "overview": r["overview"],
            })

    # ---- 9.8 输出结果 CSV ----
    print("\n" + "=" * 60)
    print(">>> 步骤 3：输出补全后的 CSV 文件")
    print("=" * 60)

    result_df = pd.DataFrame(enriched_data, columns=[
        "movieId", "title", "poster_local_path", "overview"
    ])
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  结果已保存至: '{OUTPUT_CSV}'")
    print(f"  共 {len(result_df)} 条记录")

    # ---- 9.9 汇总报告 ----
    elapsed_time = time.time() - start_time
    _, success_api, success_img, fail_api = progress_counter.snapshot()

    print("\n" + "=" * 60)
    print(">>> 处理完成！汇总报告")
    print("=" * 60)
    print(f"  总电影数:         {total}")
    print(f"  成功获取剧情:     {success_api}  ({success_api/total*100:.1f}%)")
    print(f"  成功下载海报:     {success_img}  ({success_img/total*100:.1f}%)")
    print(f"  请求失败:         {fail_api}  ({fail_api/total*100:.1f}%)")
    print(f"  总耗时:           {elapsed_time:.1f} 秒")
    print(f"  平均速度:         {total/elapsed_time:.1f} 部/秒")
    print(f"  输出文件:         {OUTPUT_CSV}")
    if ENABLE_IMAGE_DOWNLOAD:
        print(f"  海报目录:         {POSTER_DIR}/")
    print("=" * 60)


# ============================================================================
# [程序入口]
# ============================================================================

if __name__ == "__main__":
    main()
