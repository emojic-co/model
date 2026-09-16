package ing.emojify.model

import org.junit.Assert.assertEquals
import org.junit.Test

class NavTest {
    @Test
    fun `cycle wraps forward and backward`() {
        val list = listOf("a", "b", "c")
        assertEquals("b", cycle(list, "a", 1))
        assertEquals("a", cycle(list, "c", 1))
        assertEquals("a", cycle(list, "b", -1))
        assertEquals("c", cycle(list, "a", -1))
    }

    @Test
    fun `cycle starts at first or last when current is not in the list`() {
        val list = listOf("a", "b", "c")
        assertEquals("a", cycle(list, null, 1))
        assertEquals("c", cycle(list, null, -1))
    }

    @Test
    fun `cycle returns current when list is empty`() {
        assertEquals("x", cycle(emptyList(), "x", 1))
    }
}
