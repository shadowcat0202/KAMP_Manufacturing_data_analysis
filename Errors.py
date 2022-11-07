class CheckType_StrList(Exception):
    def __str__(self):
        return "String 또는 List 타입으로만 입력가능합니다. -문의: 이시영"

class CheckValue_Scaler(Exception):
    def __str__(self):
        return "SCALYER 타입은 standardize / normalize 만 가능합니다. -문의: 이시영"