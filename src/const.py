
mbti_p_typs = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
]

mbti_typ2idx = {
    s:i for i, s in enumerate(mbti_p_typs)
}

mbti_idx2typ = {
    i:s for i, s in enumerate(mbti_p_typs)
}

MBTI_CLASSES = len(mbti_p_typs)