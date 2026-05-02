"""Tool definitions in OpenAI function-calling JSON Schema format.

这些定义喂给 LLM,LLM 据此决策"调哪个工具、传什么参数".
不参与 HTTP 校验(那是 api/schemas.py 的工作).
"""

LEGAL_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "legal_search",
        "description": (
            "查询中国法律条文(语义检索 + 精排).适用于:\n"
            "- 概念解释(如'什么是合同违约')\n"
            "- 维权咨询(如'老板拖欠工资怎么办')\n"
            "- 法律责任分析(如'合同诈骗的责任')\n"
            "不适合:已知具体法律名 + 条款号的精确查询(请用 get_law_article)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "用户的法律问题,可以是自然语言. 可以适当扩展关键词以提升召回质量."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


GET_LAW_ARTICLE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_law_article",
        "description": (
            "按法律名 + 条款号精确查询条文原文.适用于:\n"
            "- 用户问'XX法第X条规定了什么'\n"
            "- 用户引用具体条款号问解读\n"
            "条款号必须保留中文数字格式,例如'第五百七十七条'(不是'577'或'5百77').\n"
            "法律名可以是简称('民法典')或全称('中华人民共和国民法典'),系统会自动模糊匹配."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "law_title": {
                    "type": "string",
                    "description": "法律名称,可以是简称('民法典')或全称('中华人民共和国民法典')",
                },
                "article_no": {
                    "type": "string",
                    "description": "条款号,中文数字格式,如'第五百七十七条'、'第三十条'、'第一条'",
                },
            },
            "required": ["law_title", "article_no"],
        },
    },
}


ALL_TOOLS = [LEGAL_SEARCH_TOOL, GET_LAW_ARTICLE_TOOL]


__all__ = ["LEGAL_SEARCH_TOOL", "GET_LAW_ARTICLE_TOOL", "ALL_TOOLS"]
