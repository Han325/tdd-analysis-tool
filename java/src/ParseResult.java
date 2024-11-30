// ParseResult.java
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;

public class ParseResult {
    private List<String> imports = new ArrayList<>();
    private List<String> classes = new ArrayList<>();
    private List<String> methods = new ArrayList<>();
    private Set<String> testMethods = new HashSet<>();
    private boolean isAbstract = false;
    private boolean isUtility = false;
    private boolean hasTestImports = false;
    private boolean hasTestAnnotations = false;
    
    public void addImport(String imp) { imports.add(imp); }
    public void addClass(String cls) { classes.add(cls); }
    public void addMethod(String method) { methods.add(method); }
    public void addTestMethod(String method) { testMethods.add(method); }
    public void setAbstract(boolean value) { isAbstract = value; }
    public void setUtility(boolean value) { isUtility = value; }
    public void setHasTestImports(boolean value) { hasTestImports = value; }
    public void setHasTestAnnotations(boolean value) { hasTestAnnotations = value; }
    
    public List<String> getImports() { return imports; }
    public List<String> getClasses() { return classes; }
    public List<String> getMethods() { return methods; }
    public Set<String> getTestMethods() { return testMethods; }
    public boolean isAbstract() { return isAbstract; }
    public boolean isUtility() { return isUtility; }
    public boolean hasTestImports() { return hasTestImports; }
    public boolean hasTestAnnotations() { return hasTestAnnotations; }
}