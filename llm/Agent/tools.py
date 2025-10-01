from datetime import datetime

# 获取当前日期和时间
def get_current_datetime() -> str:
    """
    获取当前日期和时间。
    :return: 当前日期和时间的字符串表示。
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def count_letter_in_string(a: str, b: str):
    """
    统计字符串中某个字母的出现次数。
    :param a: 要搜索的字符串。
    :param b: 要统计的字母。
    :return: 字母在字符串中出现的次数。
    """
    return str(a.count(b))

def search_wikipedia(query: str) -> str:
    """
    在维基百科中搜索指定查询的前三个页面摘要。
    :param query: 要搜索的查询字符串。
    :return: 包含前三个页面摘要的字符串。
    """
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:  # 取前三个页面标题
        try:
            # 使用 wikipedia 模块的 page 函数，获取指定标题的维基百科页面对象。
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            # 获取页面摘要
            summaries.append(f"页面: {page_title}\n摘要: {wiki_page.summary}")
        except (
                wikipedia.exceptions.PageError,
                wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "维基百科没有搜索到合适的结果"
    return "\n\n".join(summaries)
# ... (可能还有其他工具函数)
