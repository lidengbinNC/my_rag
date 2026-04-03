from my_rag.domain.knowledge.profiles import loads_str_list, profile_matches


def test_loads_list_handles_empty_and_json():
    assert loads_str_list(None) == []
    assert loads_str_list('["faq", "policy"]') == ["faq", "policy"]


def test_profile_matches_domain_tag_and_role_filters():
    assert profile_matches(
        profile_domain="policy",
        profile_visibility="internal",
        profile_language="en-US",
        tags_json='["refund", "delay"]',
        allowed_roles_json='["agent", "admin"]',
        domain="policy",
        tag="refund",
        role="agent",
    ) is True
    assert profile_matches(
        profile_domain="policy",
        profile_visibility="internal",
        profile_language="en-US",
        tags_json='["refund", "delay"]',
        allowed_roles_json='["agent", "admin"]',
        domain="faq",
    ) is False
    assert profile_matches(
        profile_domain="policy",
        profile_visibility="internal",
        profile_language="en-US",
        tags_json='["refund", "delay"]',
        allowed_roles_json='["agent", "admin"]',
        tag="unknown",
    ) is False
    assert profile_matches(
        profile_domain="policy",
        profile_visibility="internal",
        profile_language="en-US",
        tags_json='["refund", "delay"]',
        allowed_roles_json='["agent", "admin"]',
        role="bot",
    ) is False
