from src.crawler.parsers.article_content import extract_image_urls


def test_extract_image_urls_includes_open_graph_image_when_body_has_no_img():
    html = """
    <html>
      <head>
        <meta property="og:image" content="/images/news-main.jpg">
      </head>
      <body>
        <article id="dic_area">기사 본문만 있고 이미지 태그는 없는 경우입니다.</article>
      </body>
    </html>
    """

    assert extract_image_urls(html, "https://example.com/news/1") == [
        "https://example.com/images/news-main.jpg"
    ]


def test_extract_image_urls_includes_json_ld_image_object():
    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "@type": "NewsArticle",
          "image": {
            "@type": "ImageObject",
            "url": "https://cdn.example.com/article.webp"
          }
        }
        </script>
      </head>
      <body>
        <article id="dic_area">본문 이미지가 lazy 렌더링되는 기사입니다.</article>
      </body>
    </html>
    """

    assert extract_image_urls(html, "https://example.com/news/2") == [
        "https://cdn.example.com/article.webp"
    ]
