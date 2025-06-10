import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if the server is running."""
    print("🔹 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_ask_question(question):
    """Test asking a question to the chatbot."""
    print(f"\n🔹 Testing Question: {question}")
    
    question_data = {
        "question": question,
        "max_results": 3,
        "language": "ar"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ask", json=question_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Status: {result['status']}")
            print(f"Sources: {result['relevant_sources']}")
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

def test_list_faq():
    """Test listing all FAQ entries."""
    print("\n🔹 Testing FAQ List...")
    try:
        response = requests.get(f"{BASE_URL}/faq/list")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Total FAQ entries: {result['total']}")
            print("First 3 entries:")
            for entry in result['entries'][:3]:
                print(f"- {entry['question']}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting API Tests...")
    
    # Test 1: Health check
    if not test_health_check():
        print("❌ Server is not running. Make sure to start main.py first!")
        return
    
    # Test 2: List FAQ entries
    test_list_faq()
    
    # Test 3: Ask questions
    test_questions = [
        "كيف يمكنني إنشاء حساب جديد؟",
        "ما هي رسوم المنصة؟",
        "كيف يعمل نظام المزايدة؟",
        "سؤال غير موجود في الأسئلة الشائعة"
    ]
    
    for question in test_questions:
        test_ask_question(question)
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()